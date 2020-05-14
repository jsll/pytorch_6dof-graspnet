import torch.utils.data as data
import numpy as np
import pickle
import os
import copy
import json
from utils.sample import Object
from utils import utils
import glob
from renderer.online_object_renderer import OnlineObjectRenderer
import threading


class NoPositiveGraspsException(Exception):
    """raised when there's no positive grasps for an object."""
    pass


class BaseDataset(data.Dataset):
    def __init__(self,
                 opt,
                 caching=True,
                 min_difference_allowed=(0, 0, 0),
                 max_difference_allowed=(3, 3, 0),
                 collision_hard_neg_min_translation=(-0.03, -0.03, -0.03),
                 collision_hard_neg_max_translation=(0.03, 0.03, 0.03),
                 collision_hard_neg_min_rotation=(-0.6, -0.2, -0.6),
                 collision_hard_neg_max_rotation=(+0.6, +0.2, +0.6),
                 collision_hard_neg_num_perturbations=10):
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.mean = 0
        self.std = 1
        self.ninput_channels = None
        self.current_pc = None
        self.caching = caching
        self.cache = {}
        self.collision_hard_neg_min_translation = collision_hard_neg_min_translation
        self.collision_hard_neg_max_translation = collision_hard_neg_max_translation
        self.collision_hard_neg_min_rotation = collision_hard_neg_min_rotation
        self.collision_hard_neg_max_rotation = collision_hard_neg_max_rotation
        self.collision_hard_neg_num_perturbations = collision_hard_neg_num_perturbations
        self.lock = threading.Lock()
        for i in range(3):
            assert (collision_hard_neg_min_rotation[i] <=
                    collision_hard_neg_max_rotation[i])
            assert (collision_hard_neg_min_translation[i] <=
                    collision_hard_neg_max_translation[i])

        self.renderer = OnlineObjectRenderer(caching=True)

        if opt.use_uniform_quaternions:
            self.all_poses = utils.uniform_quaternions()
        else:
            self.all_poses = utils.nonuniform_quaternions()

        self.eval_files = [
            json.load(open(f)) for f in glob.glob(
                os.path.join(self.opt.dataset_root_folder, 'splits', '*.json'))
        ]

    def apply_dropout(self, pc):
        if self.opt.occlusion_nclusters == 0 or self.opt.occlusion_dropout_rate == 0.:
            return np.copy(pc)

        labels = utils.farthest_points(pc, self.opt.occlusion_nclusters,
                                       utils.distance_by_translation_point)

        removed_labels = np.unique(labels)
        removed_labels = removed_labels[np.random.rand(removed_labels.shape[0])
                                        < self.opt.occlusion_dropout_rate]
        if removed_labels.shape[0] == 0:
            return np.copy(pc)
        mask = np.ones(labels.shape, labels.dtype)
        for l in removed_labels:
            mask = np.logical_and(mask, labels != l)
        return pc[mask]

    def render_random_scene(self, camera_pose=None):
        """
          Renders a random view and return (pc, camera_pose, object_pose). 
          object_pose is None for single object per scene.
        """
        if camera_pose is None:
            viewing_index = np.random.randint(0, high=len(self.all_poses))
            camera_pose = self.all_poses[viewing_index]

        in_camera_pose = copy.deepcopy(camera_pose)
        _, _, pc, camera_pose = self.renderer.render(in_camera_pose)
        pc = self.apply_dropout(pc)
        pc = utils.regularize_pc_point_count(pc, self.opt.npoints)
        pc_mean = np.mean(pc, 0, keepdims=True)
        pc[:, :3] -= pc_mean[:, :3]
        camera_pose[:3, 3] -= pc_mean[0, :3]

        return pc, camera_pose, in_camera_pose

    def change_object_and_render(self,
                                 cad_path,
                                 cad_scale,
                                 camera_pose=None,
                                 thread_id=0):
        if camera_pose is None:
            viewing_index = np.random.randint(0, high=len(self.all_poses))
            camera_pose = self.all_poses[viewing_index]

        in_camera_pose = copy.deepcopy(camera_pose)
        _, _, pc, camera_pose = self.renderer.change_and_render(
            cad_path, cad_scale, in_camera_pose, thread_id)
        pc = self.apply_dropout(pc)
        pc = utils.regularize_pc_point_count(pc, self.opt.npoints)
        pc_mean = np.mean(pc, 0, keepdims=True)
        pc[:, :3] -= pc_mean[:, :3]
        camera_pose[:3, 3] -= pc_mean[0, :3]

        return pc, camera_pose, in_camera_pose

    def change_object(self, cad_path, cad_scale):
        self.renderer.change_object(cad_path, cad_scale)

    def read_grasp_file(self, path, return_all_grasps=False):
        file_name = path
        if self.caching and file_name in self.cache:
            pos_grasps, pos_qualities, neg_grasps, neg_qualities, cad, cad_path, cad_scale = copy.deepcopy(
                self.cache[file_name])
            return pos_grasps, pos_qualities, neg_grasps, neg_qualities, cad, cad_path, cad_scale

        pos_grasps, pos_qualities, neg_grasps, neg_qualities, cad, cad_path, cad_scale = self.read_object_grasp_data(
            path,
            ratio_of_grasps_to_be_used=self.opt.grasps_ratio,
            return_all_grasps=return_all_grasps)

        if self.caching:
            self.cache[file_name] = (pos_grasps, pos_qualities, neg_grasps,
                                     neg_qualities, cad, cad_path, cad_scale)
            return copy.deepcopy(self.cache[file_name])

        return pos_grasps, pos_qualities, neg_grasps, neg_qualities, cad, cad_path, cad_scale

    def read_object_grasp_data(self,
                               json_path,
                               quality='quality_flex_object_in_gripper',
                               ratio_of_grasps_to_be_used=1.,
                               return_all_grasps=False):
        """
        Reads the grasps from the json path and loads the mesh and all the 
        grasps.
        """
        num_clusters = self.opt.num_grasp_clusters
        root_folder = self.opt.dataset_root_folder

        if num_clusters <= 0:
            raise NoPositiveGraspsException

        json_dict = json.load(open(json_path))

        object_model = Object(os.path.join(root_folder, json_dict['object']))
        object_model.rescale(json_dict['object_scale'])
        object_model = object_model.mesh
        object_mean = np.mean(object_model.vertices, 0, keepdims=1)

        object_model.vertices -= object_mean
        grasps = np.asarray(json_dict['transforms'])
        grasps[:, :3, 3] -= object_mean

        flex_qualities = np.asarray(json_dict[quality])
        try:
            heuristic_qualities = np.asarray(
                json_dict['quality_number_of_contacts'])
        except KeyError:
            heuristic_qualities = np.ones(flex_qualities.shape)

        successful_mask = np.logical_and(flex_qualities > 0.01,
                                         heuristic_qualities > 0.01)

        positive_grasp_indexes = np.where(successful_mask)[0]
        negative_grasp_indexes = np.where(~successful_mask)[0]

        positive_grasps = grasps[positive_grasp_indexes, :, :]
        negative_grasps = grasps[negative_grasp_indexes, :, :]
        positive_qualities = heuristic_qualities[positive_grasp_indexes]
        negative_qualities = heuristic_qualities[negative_grasp_indexes]

        def cluster_grasps(grasps, qualities):
            cluster_indexes = np.asarray(
                utils.farthest_points(grasps, num_clusters,
                                      utils.distance_by_translation_grasp))
            output_grasps = []
            output_qualities = []

            for i in range(num_clusters):
                indexes = np.where(cluster_indexes == i)[0]
                if ratio_of_grasps_to_be_used < 1:
                    num_grasps_to_choose = max(
                        1,
                        int(ratio_of_grasps_to_be_used * float(len(indexes))))
                    if len(indexes) == 0:
                        raise NoPositiveGraspsException
                    indexes = np.random.choice(indexes,
                                               size=num_grasps_to_choose,
                                               replace=False)

                output_grasps.append(grasps[indexes, :, :])
                output_qualities.append(qualities[indexes])

            output_grasps = np.asarray(output_grasps)
            output_qualities = np.asarray(output_qualities)

            return output_grasps, output_qualities

        if not return_all_grasps:
            positive_grasps, positive_qualities = cluster_grasps(
                positive_grasps, positive_qualities)
            negative_grasps, negative_qualities = cluster_grasps(
                negative_grasps, negative_qualities)
            num_positive_grasps = np.sum([p.shape[0] for p in positive_grasps])
            num_negative_grasps = np.sum([p.shape[0] for p in negative_grasps])
        else:
            num_positive_grasps = positive_grasps.shape[0]
            num_negative_grasps = negative_grasps.shape[0]
        return positive_grasps, positive_qualities, negative_grasps, negative_qualities, object_model, os.path.join(
            root_folder, json_dict['object']), json_dict['object_scale']

    def sample_grasp_indexes(self, n, grasps, qualities):
        """
          Stratified sampling of the grasps.
        """
        nonzero_rows = [i for i in range(len(grasps)) if len(grasps[i]) > 0]
        num_clusters = len(nonzero_rows)
        replace = n > num_clusters
        if num_clusters == 0:
            raise NoPositiveGraspsException

        grasp_rows = np.random.choice(range(num_clusters),
                                      size=n,
                                      replace=replace).astype(np.int32)
        grasp_rows = [nonzero_rows[i] for i in grasp_rows]
        grasp_cols = []
        for grasp_row in grasp_rows:
            if len(grasps[grasp_rows]) == 0:
                raise ValueError('grasps cannot be empty')

            grasp_cols.append(np.random.randint(len(grasps[grasp_row])))

        grasp_cols = np.asarray(grasp_cols, dtype=np.int32)

        return np.vstack((grasp_rows, grasp_cols)).T

    def get_mean_std(self):
        """ Computes Mean and Standard Deviation from Training Data
        If mean/std file doesn't exist, will compute one
        :returns
        mean: N-dimensional mean
        std: N-dimensional standard deviation
        ninput_channels: N
        (here N=5)
        """

        mean_std_cache = os.path.join(self.opt.dataset_root_folder,
                                      'mean_std_cache.p')
        if not os.path.isfile(mean_std_cache):
            print('computing mean std from train data...')
            # doesn't run augmentation during m/std computation
            num_aug = self.opt.num_aug
            self.opt.num_aug = 1
            mean, std = np.array(0), np.array(0)
            for i, data in enumerate(self):
                if i % 500 == 0:
                    print('{} of {}'.format(i, self.size))
                features = data['edge_features']
                mean = mean + features.mean(axis=1)
                std = std + features.std(axis=1)
            mean = mean / (i + 1)
            std = std / (i + 1)
            transform_dict = {
                'mean': mean[:, np.newaxis],
                'std': std[:, np.newaxis],
                'ninput_channels': len(mean)
            }
            with open(mean_std_cache, 'wb') as f:
                pickle.dump(transform_dict, f)
            print('saved: ', mean_std_cache)
            self.opt.num_aug = num_aug
        # open mean / std from file
        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)
            print('loaded mean / std from cache')
            self.mean = transform_dict['mean']
            self.std = transform_dict['std']
            self.ninput_channels = transform_dict['ninput_channels']

    def make_dataset(self):
        split_files = os.listdir(
            os.path.join(self.opt.dataset_root_folder,
                         self.opt.splits_folder_name))
        files = []
        for split_file in split_files:
            if split_file.find('.json') < 0:
                continue
            should_go_through = False
            if self.opt.allowed_categories == '':
                should_go_through = True
                if self.opt.blacklisted_categories != '':
                    if self.opt.blacklisted_categories.find(
                            split_file[:-5]) >= 0:
                        should_go_through = False
            else:
                if self.opt.allowed_categories.find(split_file[:-5]) >= 0:
                    should_go_through = True

            if should_go_through:
                files += [
                    os.path.join(self.opt.dataset_root_folder,
                                 self.opt.grasps_folder_name, f)
                    for f in json.load(
                        open(
                            os.path.join(self.opt.dataset_root_folder,
                                         self.opt.splits_folder_name,
                                         split_file)))[self.opt.dataset_split]
                ]
        return files


def collate_fn(batch):
    """Creates mini-batch tensors
    We should build custom collate_fn rather than using default collate_fn
    """
    batch = list(filter(lambda x: x is not None, batch))  #
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        meta.update({key: np.concatenate([d[key] for d in batch])})
    return meta
