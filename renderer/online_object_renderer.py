# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import copy
import cv2
import h5py
import utils.sample as sample
import utils.utils as utils
import math
import sys
import argparse
import os
# Uncomment following line for headless rendering
# os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender

import trimesh
import trimesh.transformations as tra
import multiprocessing as mp

# Cover python 2.7 and python 3.5
try:
    import Queue
except:
    import queue as Queue


class OnlineObjectRendererMultiProcess(mp.Process):
    def __init__(self, caching=True):
        mp.Process.__init__(self)
        self._renderer = None
        self._caching = caching
        self._queue = mp.Queue()
        self._output_queue = mp.Queue()
        self._should_stop = False

    def run(self):
        self._renderer = OnlineObjectRenderer(caching=self._caching)
        while not self._should_stop:
            try:
                request = self._queue.get(timeout=1)
            except Queue.Empty:
                continue
            if request[0] == 'render':
                self._process_render_request(request)
            elif request[0] == 'change_object':
                self._process_change_object_request(request)
            else:
                raise ValueError('unknown requst', request)

    def _process_render_request(self, render_request):
        pose = render_request[1]
        try:
            output = self._renderer.render(pose)
            self._output_queue.put(('ok', output))
        except Exception as e:
            self._output_queue.put(('no', str(e)))

    def _process_change_object_request(self, change_object_request):
        #print('change object', change_object_request)
        cad_path = change_object_request[1]
        cad_scale = change_object_request[2]
        try:
            self._renderer.change_object(cad_path, cad_scale)
            self._output_queue.put(('ok', ))
        except Exception as e:
            self._output_queue.put(('no', str(e)))

    def render(self, pose):
        self._queue.put(('render', pose))

        outcome = self._output_queue.get(timeout=10)

        if outcome[0] != 'ok':
            print('------------->', outcome)
            raise ValueError(outcome[1])
        else:
            print(type(self._renderer))
            print(len(outcome))
            if len(outcome) == 1:
                print(outcome)
                input()
            return outcome[1]

    def change_object(self, cad_path, cad_scale):
        self._queue.put(('change_object', cad_path, cad_scale))

        outcome = self._output_queue.get(timeout=10)
        if outcome[0] != 'ok':
            raise ValueError(outcome[1])


class OnlineObjectRenderer:
    def __init__(self, fov=np.pi / 6, caching=True):
        """
        Args:
          fov: float, 
        """
        self._fov = fov
        self.mesh = None
        self._scene = None
        self.tmesh = None
        self._init_scene()
        self._current_context = None
        self._cache = {} if caching else None
        self._caching = caching

    def _init_scene(self):
        self._scene = pyrender.Scene()
        camera = pyrender.PerspectiveCamera(
            yfov=self._fov, aspectRatio=1.0,
            znear=0.001)  # do not change aspect ratio
        camera_pose = tra.euler_matrix(np.pi, 0, 0)

        self._scene.add(camera, pose=camera_pose, name='camera')

        #light = pyrender.SpotLight(color=np.ones(4), intensity=3., innerConeAngle=np.pi/16, outerConeAngle=np.pi/6.0)
        #self._scene.add(light, pose=camera_pose, name='light')

        self.renderer = pyrender.OffscreenRenderer(400, 400)

    def _load_object(self, path, scale):
        if (path, scale) in self._cache:
            return self._cache[(path, scale)]
        obj = sample.Object(path)
        obj.rescale(scale)

        tmesh = obj.mesh
        tmesh_mean = np.mean(tmesh.vertices, 0)
        tmesh.vertices -= np.expand_dims(tmesh_mean, 0)

        lbs = np.min(tmesh.vertices, 0)
        ubs = np.max(tmesh.vertices, 0)
        object_distance = np.max(ubs - lbs) * 5

        mesh = pyrender.Mesh.from_trimesh(tmesh)

        context = {
            'tmesh': copy.deepcopy(tmesh),
            'distance': object_distance,
            'node': pyrender.Node(mesh=mesh),
            'mesh_mean': np.expand_dims(tmesh_mean, 0),
        }

        self._cache[(path, scale)] = context
        return self._cache[(path, scale)]

    def change_object(self, path, scale):
        if self._current_context is not None:
            self._scene.remove_node(self._current_context['node'])

        if not self._caching:
            self._cache = {}

        self._current_context = self._load_object(path, scale)
        self._scene.add_node(self._current_context['node'])

    def current_context(self):
        return self._current_context

    def _to_pointcloud(self, depth):
        fy = fx = 0.5 / np.tan(self._fov * 0.5)  # aspectRatio is one.
        height = depth.shape[0]
        width = depth.shape[1]

        mask = np.where(depth > 0)

        x = mask[1]
        y = mask[0]

        normalized_x = (x.astype(np.float32) - width * 0.5) / width
        normalized_y = (y.astype(np.float32) - height * 0.5) / height

        world_x = normalized_x * depth[y, x] / fx
        world_y = normalized_y * depth[y, x] / fy
        world_z = depth[y, x]
        ones = np.ones(world_z.shape[0], dtype=np.float32)

        return np.vstack((world_x, world_y, world_z, ones)).T

    def render(self, pose, render_pc=True):
        if self._current_context is None:
            raise ValueError('invoke change_object first')

        transferred_pose = pose.copy()
        transferred_pose[2, 3] = self._current_context['distance']
        self._scene.set_pose(self._current_context['node'], transferred_pose)

        color, depth = self.renderer.render(self._scene)

        if render_pc:
            pc = self._to_pointcloud(depth)
        else:
            pc = None

        return color, depth, pc, transferred_pose

    def render_canonical_pc(self, poses):
        all_pcs = []
        for pose in poses:
            _, _, pc, pose = self.render(pose)
            pc = pc.dot(utils.inverse_transform(pose).T)
            all_pcs.append(pc)
        all_pcs = np.concatenate(all_pcs, 0)
        return all_pcs

    def start(self):
        pass

    def join(self):
        pass

    def terminate(self):
        pass
