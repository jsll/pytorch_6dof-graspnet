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
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = self.make_dataset(self.dir)
        self.size = len(self.paths)
        self.get_mean_std()
        opt.input_nc = self.ninput_channels

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

        self.change_object(cad_path, cad_scale)
        pc, camera_pose, _ = self.render_random_scene()
        output_qualities = []
        output_grasps = []

        for iter in range(self.opt.batch_size):
            selected_grasp_index = all_clusters[iter]

            selected_grasp = pos_grasps[selected_grasp_index[0]][
                selected_grasp_index[1]]
            selected_quality = pos_qualities[selected_grasp_index[0]][
                selected_grasp_index[1]]
            output_qualities.append(selected_quality)
            output_grasps.append(camera_pose.dot(selected_grasp))

        gt_control_points = utils.transform_control_points(
            output_grasps, self.opt.num_grasps_per_object, mode='rt')

        meta['pc'] = np.array([pc] * self.opt.num_grasps_per_object)
        meta['grasp_rt'] = np.array(output_grasps)
        meta['pc_pose'] = np.array([utils.inverse_transform(camera_pose)] *
                                   self.opt.num_grasps_per_object)
        meta['cad_path'] = np.array([cad_path] *
                                    self.opt.num_grasps_per_object)
        meta['cad_scale'] = np.array([cad_scale] *
                                     self.opt.num_grasps_per_object)
        meta['quality'] = np.array(output_qualities)
        meta['target_cps'] = np.array(gt_control_points)
        return meta

    def __len__(self):
        return self.size

    @staticmethod
    def make_dataset(path):
        grasps = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                path = os.path.join(root, fname)
                grasps.append(path)

        return grasps
