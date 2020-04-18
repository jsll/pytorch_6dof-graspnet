import argparse
import os
from util import util
import torch


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # data params
        self.parser.add_argument(
            '--dataroot',
            required=True,
            help='path to meshes (should have subfolders train, test)')
        self.parser.add_argument('--num_objects_per_batch', type=int, default=1, help='data batch size.')
        self.parser.add_argument('--num_grasps_per_object', type=int, default=64)
        self.parser.add_argument('--npoints', type=int, default=1024, help='number of points in each batch')
        self.parser.add_argument('--occlusion_nclusters', type=int, default=0, help='clusters the points to nclusters to be selected for simulating the dropout')
        self.parser.add_argument('--occlusion_dropout_rate', type=float, default=0,
                        help='probability at which the clusters are removed from point cloud.')
        self.parser.add_argument('--depth_noise', type=float, default=0.0)  # to be used in the data reader.
        self.parser.add_argument('--num_grasp_clusters', type=int, default=32)
        self.parser.add_argument('--arch',
                                 choices={"vae", "gan", "evaluator"},
                                 default='vae')
        self.parser.add_argument('--max_dataset_size',
                                 type=int,
                                 default=float("inf"),
                                 help='Maximum number of samples per epoch')
        # network params
        self.parser.add_argument('--batch_size',
                                 type=int,
                                 default=16,
                                 help='input batch size')
        # network params
        self.parser.add_argument('--grasps_per_object',
                                 type=int,
                                 default=16,
                                 help='Grasps per object')
       self.parser.add_argument('--num_threads',
                                 default=3,
                                 type=int,
                                 help='# threads for loading data')
        self.parser.add_argument(
            '--gpu_ids',
            type=str,
            default='0',
            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument(
            '--name',
            type=str,
            default='debug',
            help=
            'name of the experiment. It decides where to store samples and models'
        )
        self.parser.add_argument('--checkpoints_dir',
                                 type=str,
                                 default='./checkpoints',
                                 help='models are saved here')
        self.parser.add_argument(
            '--serial_batches',
            action='store_true',
            help='if true, takes meshes in order, otherwise takes them randomly'
        )
        self.parser.add_argument('--seed',
                                 type=int,
                                 help='if specified, uses seed')
        self.parser.add_argument('--training_splits', type=str, default='train',
                        help='can be any combination of train and test without any space.')
        self.parser.add_argument('--gripper', type=str, default='panda',
                        help='type of the gripper. Leave it to panda if you want to use it for franka robot')
        self.parser.add_argument('--latent_size', type=int, default=2)
        self.parser.add_argument('--gripper_pc_npoints', type=int, default=-1,
                        help='number of points representing the gripper. -1 just uses the points on the finger and also the base. other values use subsampling of the gripper mesh')
        self.parser.add_argument('--merge_pcs_in_vae_encoder', type=int, default=0,
                        help='whether to create unified pc in encoder by coloring the points (similar to evaluator')
        self.parser.add_argument('--allowed_categories', type=str, default='',
                        help='if left blank uses all the categories in the <DATASET_ROOT_PATH>/splits/<category>.json, otherwise only chooses the categories that are set.')
        self.parser.add_argument('--use_uniform_quaternions', type=int, default=0)
        self.parser.add_argument('--model_scale', type=int, default=1,
                        help='the scale of the parameters. Use scale >= 1. Scale=2 increases the number of parameters in model by 4x.')
        self.parser.add_argument('--splits_folder_name', type=str, default='splits',
                        help='Folder name for the directory that has all the jsons for train/test splits.')
        self.parser.add_argument('--grasps_folder_name', type=str, default='grasps',
                        help='Directory that contains the grasps. Will be joined with the dataset_root_folder and the file names as defined in the splits.')
        self.parser.add_argument('--pointnet_radius', help='Radius for ball query for PointNet++, just the first layer', type=float, default=0.02)
        self.parser.add_argument('--pointnet_nclusters', help='Number of cluster centroids for PointNet++, just the first layer', type=int, default=128)


    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args()
        self.opt.is_train = self.is_train  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        if self.opt.seed is not None:
            import numpy as np
            import random
            torch.manual_seed(self.opt.seed)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)

        if self.opt.export_folder:
            self.opt.export_folder = os.path.join(self.opt.checkpoints_dir,
                                                  self.opt.name,
                                                  self.opt.export_folder)
            util.mkdir(self.opt.export_folder)

        if self.is_train:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdir(expr_dir)

            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
