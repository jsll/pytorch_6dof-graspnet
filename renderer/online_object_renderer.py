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
import time
# Uncomment following line for headless rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender

import trimesh
import trimesh.transformations as tra
from multiprocessing import Manager
import multiprocessing as mp


class OnlineObjectRenderer:
    def __init__(self, fov=np.pi / 6, caching=True):
        """
        Args:
          fov: float, 
        """
        self._fov = fov
        self._fy = self._fx = 1 / (0.5 / np.tan(self._fov * 0.5)
                                   )  # aspectRatio is one.
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

        self.renderer = None

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
        height = depth.shape[0]
        width = depth.shape[1]

        mask = np.where(depth > 0)

        x = mask[1]
        y = mask[0]

        normalized_x = (x.astype(np.float32) - width * 0.5) / width
        normalized_y = (y.astype(np.float32) - height * 0.5) / height

        world_x = self._fx * normalized_x * depth[y, x]
        world_y = self._fy * normalized_y * depth[y, x]
        world_z = depth[y, x]
        ones = np.ones(world_z.shape[0], dtype=np.float32)

        return np.vstack((world_x, world_y, world_z, ones)).T

    def change_and_render(self, cad_path, cad_scale, pose, render_pc=True):
        self.change_object(cad_path, cad_scale)
        color, depth, pc, transferred_pose = self.render(pose)

        return color, depth, pc, transferred_pose

    def render(self, pose, render_pc=True):
        if self.renderer is None:
            self.renderer = pyrender.OffscreenRenderer(400, 400)
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
