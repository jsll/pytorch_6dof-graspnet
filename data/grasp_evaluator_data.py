import os
import torch
from data.base_dataset import BaseDataset, NoPositiveGraspsException
import numpy as np
from utils import utils
import random
try:
    from Queue import Queue
except:
    from queue import Queue


class GraspSamplingData(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = self.make_dataset(self.dir)
        self.size = len(self.paths)
        self.get_mean_std()
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index]
        if self.opt.balanced_data:
            data = self.get_uniform_evaluator_data(path)
        else:
            data = self.get_nonuniform_evaluator_data(path)

        meta = {}
        meta['pc'] = data[0]
        meta['grasp_rt'] = data[1]
        meta['labels'] = data[2]
        meta['quality'] = data[3]
        meta['pc_pose'] = data[4]
        meta['cad_path'] = data[5]
        meta['cad_scale'] = data[6]
        return meta

    def __len__(self):
        return self.size

    def get_uniform_evaluator_data(self, path, verify_grasps=False):
        pos_grasps, pos_qualities, neg_grasps, neg_qualities, obj_mesh, cad_path, cad_scale = self.read_grasp_file(
            path)

        output_pcs = []
        output_grasps = []
        output_qualities = []
        output_labels = []
        output_pc_poses = []
        output_cad_paths = [cad_path] * self.opt.batch_size
        output_cad_scales = np.asarray([cad_scale] * self.opt.batch_size,
                                       np.float32)

        num_positive = int(self.opt.batch_size * self.opt.ratio_positive)
        positive_clusters = self.sample_grasp_indexes(num_positive, pos_grasps,
                                                      pos_qualities)
        num_hard_negative = int(self.opt.batch_size *
                                self.opt.ratio_hardnegative)
        num_flex_negative = self.opt.batch_size - num_positive - num_hard_negative
        negative_clusters = self.sample_grasp_indexes(num_flex_negative,
                                                      neg_grasps,
                                                      neg_qualities)
        hard_neg_candidates = []
        # Fill in Positive Examples.

        for clusters, grasps, qualities in zip(
            [positive_clusters, negative_clusters], [pos_grasps, neg_grasps],
            [pos_qualities, neg_qualities]):
            for cluster in clusters:
                selected_grasp = grasps[cluster[0]][cluster[1]]
                selected_quality = qualities[cluster[0]][cluster[1]]
                hard_neg_candidates += utils.perturb_grasp(
                    selected_grasp,
                    self.collision_hard_neg_num_perturbations,
                    self.collision_hard_neg_min_translation,
                    self.collision_hard_neg_max_translation,
                    self.collision_hard_neg_min_rotation,
                    self.collision_hard_neg_max_rotation,
                )

        if verify_grasps:
            collisions, heuristic_qualities = utils.evaluate_grasps(
                output_grasps, obj_mesh)
            for computed_quality, expected_quality, g in zip(
                    heuristic_qualities, output_qualities, output_grasps):
                err = abs(computed_quality - expected_quality)
                if err > 1e-3:
                    raise ValueError(
                        'Heuristic does not match with the values from data generation {}!={}'
                        .format(computed_quality, expected_quality))

        # If queue does not have enough data, fill it up with hard negative examples from the positives.
        if path not in self._collision_hard_neg_queue or len(
                self._collision_hard_neg_queue[path]) < num_hard_negative:
            if path not in self._collision_hard_neg_queue:
                self._collision_hard_neg_queue[path] = []
            #hard negatives are perturbations of correct grasps.
            collisions, heuristic_qualities = utils.evaluate_grasps(
                hard_neg_candidates, obj_mesh)

            hard_neg_mask = collisions | (heuristic_qualities < 0.001)
            hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()
            np.random.shuffle(hard_neg_indexes)
            for index in hard_neg_indexes:
                self._collision_hard_neg_queue[path].append(
                    (hard_neg_candidates[index], -1.0))
            random.shuffle(self._collision_hard_neg_queue[path])

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
            grasp, quality = self._collision_hard_neg_queue[path][i]
            output_grasps.append(grasp)
            output_qualities.append(quality)
            output_labels.append(0)

        self._collision_hard_neg_queue[path] = self._collision_hard_neg_queue[
            path][num_hard_negative:]

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
                output_pc_poses.append(utils.inverse_transform(camera_pose))

            output_grasps[iter] = camera_pose.dot(output_grasps[iter])

        output_pcs = np.asarray(output_pcs, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_labels = np.asarray(output_labels, dtype=np.int32)
        output_qualities = np.asarray(output_qualities, dtype=np.float32)
        output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)

        return output_pcs, output_grasps, output_labels, output_qualities, output_pc_poses, output_cad_paths, output_cad_scales

    @staticmethod
    def make_dataset(path):
        grasps = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                path = os.path.join(root, fname)
                grasps.append(path)

        return grasps

    def get_nonuniform_evaluator_data(self, path):

        pos_grasps, pos_qualities, neg_grasps, neg_qualities, obj_mesh, cad_path, cad_scale = self.read_grasp_file(
            path)

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
            hard_neg_candidates += utils.perturb_grasp(
                selected_grasp,
                self._collision_hard_neg_num_perturbations,
                self._collision_hard_neg_min_translation,
                self._collision_hard_neg_max_translation,
                self._collision_hard_neg_min_rotation,
                self._collision_hard_neg_max_rotation,
            )

        if self.ops.verify_grasps:
            collisions, heuristic_qualities = utils.evaluate_grasps(
                output_grasps, obj_mesh)
            for computed_quality, expected_quality, g in zip(
                    heuristic_qualities, output_qualities, output_grasps):
                err = abs(computed_quality - expected_quality)
                if err > 1e-3:
                    raise ValueError(
                        'Heuristic does not match with the values from data generation {}!={}'
                        .format(computed_quality, expected_quality))

        # If queue does not have enough data, fill it up with hard negative examples from the positives.
        if path not in self._collision_hard_neg_queue or self._collision_hard_neg_queue[
                path].qsize() < num_negative:
            if path not in self._collision_hard_neg_queue:
                self._collision_hard_neg_queue[path] = Queue()
            #hard negatives are perturbations of correct grasps.
            random_selector = np.random.rand()
            if random_selector < self._ratio_hardnegative:
                print('add hard neg')
                collisions, heuristic_qualities = utils.evaluate_grasps(
                    hard_neg_candidates, obj_mesh)

                hard_neg_mask = collisions | (heuristic_qualities < 0.001)
                hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()
                np.random.shuffle(hard_neg_indexes)
                for index in hard_neg_indexes:
                    self._collision_hard_neg_queue[path].put(
                        (hard_neg_candidates[index], -1.0))
            if random_selector >= self._ratio_hardnegative or self._collision_hard_neg_queue[
                    path].qsize() < num_negative:
                for negative_cluster in negative_clusters:
                    selected_grasp = neg_grasps[negative_cluster[0]][
                        negative_cluster[1]]
                    selected_quality = neg_qualities[negative_cluster[0]][
                        negative_cluster[1]]
                    self._collision_hard_neg_queue[path].put(
                        (selected_grasp, selected_quality))

        # Use negative examples from queue.
        for _ in range(num_negative):
            #print('qsize = ', self._collision_hard_neg_queue[file_path].qsize())
            grasp, quality = self._collision_hard_neg_queue[path].get()
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
                output_pc_poses.append(utils.inverse_transform(camera_pose))

            output_grasps[iter] = camera_pose.dot(output_grasps[iter])

        output_pcs = np.asarray(output_pcs, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_labels = np.asarray(output_labels, dtype=np.int32)
        output_qualities = np.asarray(output_qualities, dtype=np.float32)
        output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)

        return output_pcs, output_grasps, output_labels, output_qualities, output_pc_poses, output_cad_paths, output_cad_scales
