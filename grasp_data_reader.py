# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function
from __future__ import absolute_import

import argparse
import h5py
import numpy as np
import copy
import os
import math
import time
import trimesh.transformations as tra
import json
import sample
try:
    from Queue import Queue
except:
    from queue import Queue

import tensorflow as tf
from online_object_renderer import OnlineObjectRendererMultiProcess, OnlineObjectRenderer
import random
import glob


class NoPositiveGraspsException(Exception):
    """raised when there's no positive grasps for an object."""
    pass


class PointCloudReader:
    def __init__(
            self,
            root_folder,
            batch_size,
            num_grasp_clusters,
            npoints,
            min_difference_allowed=(0, 0, 0),
            max_difference_allowed=(3, 3, 0),
            occlusion_nclusters=0,
            occlusion_dropout_rate=0.,
            caching=True,
            run_in_another_process=True,
            collision_hard_neg_min_translation=(-0.03, -0.03, -0.03),
            collision_hard_neg_max_translation=(0.03, 0.03, 0.03),
            collision_hard_neg_min_rotation=(-0.6, -0.2, -0.6),
            collision_hard_neg_max_rotation=(+0.6, +0.2, +0.6),
            collision_hard_neg_num_perturbations=10,
            use_uniform_quaternions=False,
            ratio_of_grasps_used=1.0,
            ratio_positive=0.3,
            ratio_hardnegative=0.4,
            balanced_data=True,
    ):
        self._root_folder = root_folder
        self._batch_size = batch_size
        self._num_grasp_clusters = num_grasp_clusters
        self._max_difference_allowed = max_difference_allowed
        self._min_difference_allowed = min_difference_allowed
        self._npoints = npoints
        self._occlusion_nclusters = occlusion_nclusters
        self._occlusion_dropout_rate = occlusion_dropout_rate
        self._caching = caching
        self._collision_hard_neg_min_translation = collision_hard_neg_min_translation
        self._collision_hard_neg_max_translation = collision_hard_neg_max_translation
        self._collision_hard_neg_min_rotation = collision_hard_neg_min_rotation
        self._collision_hard_neg_max_rotation = collision_hard_neg_max_rotation
        self._collision_hard_neg_num_perturbations = collision_hard_neg_num_perturbations
        self._collision_hard_neg_queue = {}
        self._ratio_of_grasps_used = ratio_of_grasps_used
        self._ratio_positive = ratio_positive
        self._ratio_hardnegative = ratio_hardnegative
        self._balanced_data = balanced_data

        for i in range(3):
            assert (collision_hard_neg_min_rotation[i] <=
                    collision_hard_neg_max_rotation[i])
            assert (collision_hard_neg_min_translation[i] <=
                    collision_hard_neg_max_translation[i])

        self._current_pc = None
        self._cache = {}
        if run_in_another_process:
            self._renderer = OnlineObjectRendererMultiProcess(caching=True)
        else:
            self._renderer = OnlineObjectRenderer(caching=True)

        self._renderer.start()

        if use_uniform_quaternions:
            quaternions = [
                l[:-1].split('\t') for l in open(
                    'uniform_quaternions/data2_4608.qua', 'r').readlines()
            ]

            quaternions = [[
                float(t[0]),
                float(t[1]),
                float(t[2]),
                float(t[3])
            ] for t in quaternions]
            quaternions = np.asarray(quaternions)
            quaternions = np.roll(quaternions, 1, axis=1)
            self._all_poses = [tra.quaternion_matrix(q) for q in quaternions]
        else:
            self._all_poses = []
            for az in np.linspace(0, np.pi * 2, 30):
                for el in np.linspace(-np.pi / 2, np.pi / 2, 30):
                    self._all_poses.append(tra.euler_matrix(el, az, 0))

        self._eval_files = [
            json.load(open(f)) for f in glob.glob(
                os.path.join(self._root_folder, 'splits', '*.json'))
        ]

    def render_random_scene(self, camera_pose=None):
        """
          Renders a random view and return (pc, camera_pose, object_pose). 
          object_pose is None for single object per scene.
        """
        if camera_pose is None:
            viewing_index = np.random.randint(0, high=len(self._all_poses))
            camera_pose = self._all_poses[viewing_index]

        in_camera_pose = copy.deepcopy(camera_pose)
        _, _, pc, camera_pose = self._renderer.render(in_camera_pose)
        pc = self.apply_dropout(pc)
        pc = regularize_pc_point_count(pc, self._npoints)
        pc_mean = np.mean(pc, 0, keepdims=True)
        pc[:, :3] -= pc_mean[:, :3]
        camera_pose[:3, 3] -= pc_mean[0, :3]

        return pc, camera_pose, in_camera_pose

    def change_object(self, cad_path, cad_scale):
        self._renderer.change_object(cad_path, cad_scale)

    def get_evaluator_data(self, grasp_path, verify_grasps=False):
        if self._balanced_data:
            return self._get_uniform_evaluator_data(grasp_path, verify_grasps)

        pos_grasps, pos_qualities, neg_grasps, neg_qualities, obj_mesh, cad_path, cad_scale = self.read_grasp_file(
            grasp_path)

        output_pcs = []
        output_grasps = []
        output_qualities = []
        output_labels = []
        output_pc_poses = []
        output_cad_paths = [cad_path] * self._batch_size
        output_cad_scales = np.asarray([cad_scale] * self._batch_size,
                                       np.float32)

        num_positive = int(self._batch_size * self._ratio_positive)
        positive_clusters = self.sample_grasp_indexes(num_positive, pos_grasps,
                                                      pos_qualities)
        num_negative = self._batch_size - num_positive
        negative_clusters = self.sample_grasp_indexes(
            self._batch_size - num_positive, neg_grasps, neg_qualities)

        hard_neg_candidates = []
        # Fill in Positive Examples.
        for positive_cluster in positive_clusters:
            #print(positive_cluster)
            selected_grasp = pos_grasps[positive_cluster[0]][
                positive_cluster[1]]
            selected_quality = pos_qualities[positive_cluster[0]][
                positive_cluster[1]]
            output_grasps.append(selected_grasp)
            output_qualities.append(selected_quality)
            output_labels.append(1)
            hard_neg_candidates += perturb_grasp(
                selected_grasp,
                self._collision_hard_neg_num_perturbations,
                self._collision_hard_neg_min_translation,
                self._collision_hard_neg_max_translation,
                self._collision_hard_neg_min_rotation,
                self._collision_hard_neg_max_rotation,
            )

        if verify_grasps:
            collisions, heuristic_qualities = evaluate_grasps(
                output_grasps, obj_mesh)
            for computed_quality, expected_quality, g in zip(
                    heuristic_qualities, output_qualities, output_grasps):
                err = abs(computed_quality - expected_quality)
                if err > 1e-3:
                    raise ValueError(
                        'Heuristic does not match with the values from data generation {}!={}'
                        .format(computed_quality, expected_quality))

        # If queue does not have enough data, fill it up with hard negative examples from the positives.
        if grasp_path not in self._collision_hard_neg_queue or self._collision_hard_neg_queue[
                grasp_path].qsize() < num_negative:
            if grasp_path not in self._collision_hard_neg_queue:
                self._collision_hard_neg_queue[grasp_path] = Queue()
            #hard negatives are perturbations of correct grasps.
            random_selector = np.random.rand()
            if random_selector < self._ratio_hardnegative:
                print('add hard neg')
                collisions, heuristic_qualities = evaluate_grasps(
                    hard_neg_candidates, obj_mesh)

                hard_neg_mask = collisions | (heuristic_qualities < 0.001)
                hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()
                np.random.shuffle(hard_neg_indexes)
                for index in hard_neg_indexes:
                    self._collision_hard_neg_queue[grasp_path].put(
                        (hard_neg_candidates[index], -1.0))
            if random_selector >= self._ratio_hardnegative or self._collision_hard_neg_queue[
                    grasp_path].qsize() < num_negative:
                for negative_cluster in negative_clusters:
                    selected_grasp = neg_grasps[negative_cluster[0]][
                        negative_cluster[1]]
                    selected_quality = neg_qualities[negative_cluster[0]][
                        negative_cluster[1]]
                    self._collision_hard_neg_queue[grasp_path].put(
                        (selected_grasp, selected_quality))

        # Use negative examples from queue.
        for _ in range(num_negative):
            #print('qsize = ', self._collision_hard_neg_queue[file_path].qsize())
            grasp, quality = self._collision_hard_neg_queue[grasp_path].get()
            output_grasps.append(grasp)
            output_qualities.append(quality)
            output_labels.append(0)

        self.change_object(cad_path, cad_scale)
        for iter in range(self._batch_size):
            if iter > 0:
                output_pcs.append(np.copy(output_pcs[0]))
                output_pc_poses.append(np.copy(output_pc_poses[0]))
            else:
                pc, camera_pose, _ = self.render_random_scene()
                output_pcs.append(pc)
                output_pc_poses.append(inverse_transform(camera_pose))

            output_grasps[iter] = camera_pose.dot(output_grasps[iter])

        output_pcs = np.asarray(output_pcs, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_labels = np.asarray(output_labels, dtype=np.int32)
        output_qualities = np.asarray(output_qualities, dtype=np.float32)
        output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)

        return output_pcs, output_grasps, output_labels, output_qualities, output_pc_poses, output_cad_paths, output_cad_scales

    def _get_uniform_evaluator_data(self, grasp_path, verify_grasps=False):
        pos_grasps, pos_qualities, neg_grasps, neg_qualities, obj_mesh, cad_path, cad_scale = self.read_grasp_file(
            grasp_path)

        output_pcs = []
        output_grasps = []
        output_qualities = []
        output_labels = []
        output_pc_poses = []
        output_cad_paths = [cad_path] * self._batch_size
        output_cad_scales = np.asarray([cad_scale] * self._batch_size,
                                       np.float32)

        num_positive = int(self._batch_size * self._ratio_positive)
        positive_clusters = self.sample_grasp_indexes(num_positive, pos_grasps,
                                                      pos_qualities)
        num_hard_negative = int(self._batch_size * self._ratio_hardnegative)
        num_flex_negative = self._batch_size - num_positive - num_hard_negative
        negative_clusters = self.sample_grasp_indexes(num_flex_negative,
                                                      neg_grasps,
                                                      neg_qualities)
        #print(
        #    'positive = {}, hard_neg = {}, flex_neg = {}'.format(
        #        num_positive, num_hard_negative, num_flex_negative)
        #)

        hard_neg_candidates = []
        # Fill in Positive Examples.

        for clusters, grasps, qualities in zip(
            [positive_clusters, negative_clusters], [pos_grasps, neg_grasps],
            [pos_qualities, neg_qualities]):
            for cluster in clusters:
                selected_grasp = grasps[cluster[0]][cluster[1]]
                selected_quality = qualities[cluster[0]][cluster[1]]
                hard_neg_candidates += perturb_grasp(
                    selected_grasp,
                    self._collision_hard_neg_num_perturbations,
                    self._collision_hard_neg_min_translation,
                    self._collision_hard_neg_max_translation,
                    self._collision_hard_neg_min_rotation,
                    self._collision_hard_neg_max_rotation,
                )

        if verify_grasps:
            collisions, heuristic_qualities = evaluate_grasps(
                output_grasps, obj_mesh)
            for computed_quality, expected_quality, g in zip(
                    heuristic_qualities, output_qualities, output_grasps):
                err = abs(computed_quality - expected_quality)
                if err > 1e-3:
                    raise ValueError(
                        'Heuristic does not match with the values from data generation {}!={}'
                        .format(computed_quality, expected_quality))

        # If queue does not have enough data, fill it up with hard negative examples from the positives.
        if grasp_path not in self._collision_hard_neg_queue or len(
                self._collision_hard_neg_queue[grasp_path]
        ) < num_hard_negative:
            if grasp_path not in self._collision_hard_neg_queue:
                self._collision_hard_neg_queue[grasp_path] = []
            #hard negatives are perturbations of correct grasps.
            collisions, heuristic_qualities = evaluate_grasps(
                hard_neg_candidates, obj_mesh)

            hard_neg_mask = collisions | (heuristic_qualities < 0.001)
            hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()
            np.random.shuffle(hard_neg_indexes)
            for index in hard_neg_indexes:
                self._collision_hard_neg_queue[grasp_path].append(
                    (hard_neg_candidates[index], -1.0))
            random.shuffle(self._collision_hard_neg_queue[grasp_path])

        # Adding positive grasps
        for positive_cluster in positive_clusters:
            #print(positive_cluster)
            selected_grasp = pos_grasps[positive_cluster[0]][
                positive_cluster[1]]
            selected_quality = pos_qualities[positive_cluster[0]][
                positive_cluster[1]]
            output_grasps.append(selected_grasp)
            output_qualities.append(selected_quality)
            output_labels.append(1)

        # Adding hard neg
        for i in range(num_hard_negative):
            #print('qsize = ', self._collision_hard_neg_queue[file_path].qsize())
            grasp, quality = self._collision_hard_neg_queue[grasp_path][i]
            output_grasps.append(grasp)
            output_qualities.append(quality)
            output_labels.append(0)

        self._collision_hard_neg_queue[
            grasp_path] = self._collision_hard_neg_queue[grasp_path][
                num_hard_negative:]

        # Adding flex neg
        if len(negative_clusters) != num_flex_negative:
            raise ValueError(
                'negative clusters should have the same length as num_flex_negative {} != {}'
                .format(len(negative_clusters), num_flex_negative))

        for negative_cluster in negative_clusters:
            selected_grasp = neg_grasps[negative_cluster[0]][
                negative_cluster[1]]
            selected_quality = neg_qualities[negative_cluster[0]][
                negative_cluster[1]]
            output_grasps.append(selected_grasp)
            output_qualities.append(selected_quality)
            output_labels.append(0)

        self.change_object(cad_path, cad_scale)
        for iter in range(self._batch_size):
            if iter > 0:
                output_pcs.append(np.copy(output_pcs[0]))
                output_pc_poses.append(np.copy(output_pc_poses[0]))
            else:
                pc, camera_pose, _ = self.render_random_scene()
                output_pcs.append(pc)
                output_pc_poses.append(inverse_transform(camera_pose))

            output_grasps[iter] = camera_pose.dot(output_grasps[iter])

        output_pcs = np.asarray(output_pcs, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_labels = np.asarray(output_labels, dtype=np.int32)
        output_qualities = np.asarray(output_qualities, dtype=np.float32)
        output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)

        return output_pcs, output_grasps, output_labels, output_qualities, output_pc_poses, output_cad_paths, output_cad_scales

    def get_vae_data(self, grasp_path):
        pos_grasps, pos_qualities, _, _, _, cad_path, cad_scale = self.read_grasp_file(
            grasp_path)

        output_pcs = []
        output_grasps = []
        output_pc_poses = []
        output_cad_files = [cad_path] * self._batch_size
        output_cad_scales = np.asarray([cad_scale] * self._batch_size,
                                       dtype=np.float32)
        output_qualities = []

        all_clusters = self.sample_grasp_indexes(self._batch_size, pos_grasps,
                                                 pos_qualities)

        self.change_object(cad_path, cad_scale)
        for iter in range(self._batch_size):
            selected_grasp_index = all_clusters[iter]

            selected_grasp = pos_grasps[selected_grasp_index[0]][
                selected_grasp_index[1]]
            selected_quality = pos_qualities[selected_grasp_index[0]][
                selected_grasp_index[1]]
            output_qualities.append(selected_quality)

            if iter == 0:
                pc, camera_pose, _ = self.render_random_scene()
                output_pcs.append(pc)
                output_pc_poses.append(inverse_transform(camera_pose))
            else:
                output_pcs.append(output_pcs[0].copy())
                output_pc_poses.append(output_pc_poses[0].copy())

            output_grasps.append(camera_pose.dot(selected_grasp))

        output_pcs = np.asarray(output_pcs, dtype=np.float32)
        output_qualities = np.asarray(output_qualities, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)
        return output_pcs, output_grasps, output_pc_poses, output_cad_files, output_cad_scales, output_qualities

    def generate_object_set(self, split_name):
        obj_files = self._eval_files[np.random.randint(0, len(
            self._eval_files))][split_name]
        return os.path.join('grasps',
                            obj_files[np.random.randint(0, len(obj_files))])

    def arrange_objects(self, meshes):
        return np.eye(4)

    def __del__(self):
        print('********** terminating renderer **************')
        self._renderer.terminate()
        self._renderer.join()
        print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grasp data reader")
    parser.add_argument('--root-folder',
                        help='Root dir for data',
                        type=str,
                        default='unified_grasp_data')
    parser.add_argument('--vae-mode',
                        help='True for vae mode',
                        action='store_true',
                        default=False)
    parser.add_argument(
        '--grasps-ratio',
        help=
        'ratio of grasps to be used from each cluster. At least one grasp is chosen from each cluster.',
        type=float,
        default=1.0)
    parser.add_argument(
        '--balanced_data',
        action='store_true',
        default=False,
    )
    parser.add_argument('--allowed_category', default='', type=str)

    args = parser.parse_args()
    args.root_folder = os.path.abspath(args.root_folder)
    print('Root folder', args.root_folder)

    import glob
    from visualization_utils import draw_scene
    import mayavi.mlab as mlab

    pcreader = PointCloudReader(root_folder=args.root_folder,
                                batch_size=64,
                                num_grasp_clusters=32,
                                npoints=1024,
                                ratio_of_grasps_used=args.grasps_ratio,
                                balanced_data=args.balanced_data)

    grasp_paths = glob.glob(
        os.path.join(args.root_folder, 'grasps') + '/*.json')

    if args.allowed_category != '':
        grasp_paths = [
            g for g in grasp_paths if g.find(args.allowed_category) >= 0
        ]

    for grasp_path in grasp_paths:
        if args.vae_mode:
            output_pcs, output_grasps, output_pc_poses, output_cad_files, output_cad_scales, output_qualities = pcreader.get_vae_data(
                grasp_path)
            output_labels = None
        else:
            output_pcs, output_grasps, output_labels, output_qualities, output_pc_poses, output_cad_files, output_cad_scales = pcreader.get_evaluator_data(
                grasp_path, verify_grasps=False)

        print(output_grasps.shape)

        for pc, pose in zip(output_pcs, output_pc_poses):
            assert (np.all(pc == output_pcs[0]))
            assert (np.all(pose == output_pc_poses[0]))

        pc = output_pcs[0]
        pose = output_pc_poses[0]
        cad_file = output_cad_files[0]
        cad_scale = output_cad_scales[0]
        obj = sample.Object(cad_file)
        obj.rescale(cad_scale)
        obj = obj.mesh
        obj.vertices -= np.expand_dims(np.mean(obj.vertices, 0), 0)

        print('mean_pc', np.mean(pc, 0))
        print('pose', pose)
        draw_scene(
            pc,
            grasps=output_grasps,
            grasp_scores=None if args.vae_mode else output_labels,
        )
        mlab.figure()
        draw_scene(
            pc.dot(pose.T),
            grasps=[pose.dot(g) for g in output_grasps],
            mesh=obj,
            grasp_scores=None if args.vae_mode else output_labels,
        )
        mlab.show()
