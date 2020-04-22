import os
import torch
from data.base_dataset import BaseDataset, NoPositiveGraspsException
import numpy as np
from utils import utils


class GraspSamplingData(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataset_root_folder
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        #self.get_mean_std()
        opt.input_nc = self.ninput_channels
        self.i = 0

    def __getitem__(self, index):
        path = self.paths[index]
        pos_grasps, pos_qualities, _, _, _, cad_path, cad_scale = self.read_grasp_file(
            path)
        meta = {}
        try:
            all_clusters = self.sample_grasp_indexes(
                self.opt.num_grasps_per_object, pos_grasps, pos_qualities)
        except NoPositiveGraspsException:
            if self.opt.skip_error:
                return None
            else:
                return self.__getitem__(np.random.randint(0, self.size))

        #self.change_object(cad_path, cad_scale)
        #pc, camera_pose, _ = self.render_random_scene()
        pc, camera_pose, _ = self.change_object_and_render(
            cad_path,
            cad_scale,
            thread_id=torch.utils.data.get_worker_info().id
            if torch.utils.data.get_worker_info() else 0)

        output_qualities = []
        output_grasps = []
        for iter in range(self.opt.num_grasps_per_object):
            selected_grasp_index = all_clusters[iter]

            selected_grasp = pos_grasps[selected_grasp_index[0]][
                selected_grasp_index[1]]
            selected_quality = pos_qualities[selected_grasp_index[0]][
                selected_grasp_index[1]]
            output_qualities.append(selected_quality)
            output_grasps.append(camera_pose.dot(selected_grasp))
        gt_control_points = utils.transform_control_points_numpy(
            np.array(output_grasps), self.opt.num_grasps_per_object, mode='rt')

        meta['pc'] = np.array([pc] * self.opt.num_grasps_per_object)[:, :, :3]
        meta['grasp_rt'] = np.array(output_grasps).reshape(
            len(output_grasps), -1)

        meta['pc_pose'] = np.array([utils.inverse_transform(camera_pose)] *
                                   self.opt.num_grasps_per_object)
        meta['cad_path'] = np.array([cad_path] *
                                    self.opt.num_grasps_per_object)
        meta['cad_scale'] = np.array([cad_scale] *
                                     self.opt.num_grasps_per_object)
        meta['quality'] = np.array(output_qualities)
        meta['target_cps'] = np.array(gt_control_points[:, :, :3])
        return meta

    def __len__(self):
        return self.size