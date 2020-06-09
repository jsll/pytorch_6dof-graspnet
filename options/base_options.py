import argparse
import os
from utils import utils
import torch
import shutil
import yaml


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # data params
        self.parser.add_argument(
            '--dataset_root_folder',
            type=str,
            default=
            '/home/jens/Documents/datasets/grasping/unified_grasp_data/',
            help='path to root directory of the dataset.')
        self.parser.add_argument('--num_objects_per_batch',
                                 type=int,
                                 default=1,
                                 help='data batch size.')
        self.parser.add_argument('--num_grasps_per_object',
                                 type=int,
                                 default=64)
        self.parser.add_argument('--npoints',
                                 type=int,
                                 default=1024,
                                 help='number of points in each batch')
        self.parser.add_argument(
            '--occlusion_nclusters',
            type=int,
            default=0,
            help=
            'clusters the points to nclusters to be selected for simulating the dropout'
        )
        self.parser.add_argument(
            '--occlusion_dropout_rate',
            type=float,
            default=0,
            help=
            'probability at which the clusters are removed from point cloud.')
        self.parser.add_argument('--depth_noise', type=float,
                                 default=0.0)  # to be used in the data reader.
        self.parser.add_argument('--num_grasp_clusters', type=int, default=32)
        self.parser.add_argument('--arch',
                                 choices={"vae", "gan", "evaluator"},
                                 default='vae')
        self.parser.add_argument('--max_dataset_size',
                                 type=int,
                                 default=float("inf"),
                                 help='Maximum number of samples per epoch')
        self.parser.add_argument('--num_threads',
                                 default=3,
                                 type=int,
                                 help='# threads for loading data')
        self.parser.add_argument(
            '--gpu_ids',
            type=str,
            default='0',
            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
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
        self.parser.add_argument(
            '--gripper',
            type=str,
            default='panda',
            help=
            'type of the gripper. Leave it to panda if you want to use it for franka robot'
        )
        self.parser.add_argument('--latent_size', type=int, default=2)
        self.parser.add_argument(
            '--gripper_pc_npoints',
            type=int,
            default=-1,
            help=
            'number of points representing the gripper. -1 just uses the points on the finger and also the base. other values use subsampling of the gripper mesh'
        )
        self.parser.add_argument(
            '--merge_pcs_in_vae_encoder',
            type=int,
            default=0,
            help=
            'whether to create unified pc in encoder by coloring the points (similar to evaluator'
        )
        self.parser.add_argument(
            '--allowed_categories',
            type=str,
            default='',
            help=
            'if left blank uses all the categories in the <DATASET_ROOT_PATH>/splits/<category>.json, otherwise only chooses the categories that are set.'
        )
        self.parser.add_argument('--blacklisted_categories',
                                 type=str,
                                 default='',
                                 help='The opposite of allowed categories')

        self.parser.add_argument('--use_uniform_quaternions',
                                 type=int,
                                 default=0)
        self.parser.add_argument(
            '--model_scale',
            type=int,
            default=1,
            help=
            'the scale of the parameters. Use scale >= 1. Scale=2 increases the number of parameters in model by 4x.'
        )
        self.parser.add_argument(
            '--splits_folder_name',
            type=str,
            default='splits',
            help=
            'Folder name for the directory that has all the jsons for train/test splits.'
        )
        self.parser.add_argument(
            '--grasps_folder_name',
            type=str,
            default='grasps',
            help=
            'Directory that contains the grasps. Will be joined with the dataset_root_folder and the file names as defined in the splits.'
        )
        self.parser.add_argument(
            '--pointnet_radius',
            help='Radius for ball query for PointNet++, just the first layer',
            type=float,
            default=0.02)
        self.parser.add_argument(
            '--pointnet_nclusters',
            help=
            'Number of cluster centroids for PointNet++, just the first layer',
            type=int,
            default=128)
        self.parser.add_argument(
            '--init_type',
            type=str,
            default='normal',
            help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument(
            '--init_gain',
            type=float,
            default=0.02,
            help='scaling factor for normal, xavier and orthogonal.')
        self.parser.add_argument(
            '--grasps_ratio',
            type=float,
            default=1.0,
            help=
            'used for checking the effect of number of grasps per object on the success of the model.'
        )
        self.parser.add_argument(
            '--skip_error',
            action='store_true',
            help=
            'Will not fill the dataset with a new grasp if it raises NoPositiveGraspsException'
        )
        self.parser.add_argument(
            '--balanced_data',
            action='store_true',
            default=False,
        )
        self.parser.add_argument(
            '--confidence_weight',
            type=float,
            default=1.0,
            help=
            'initially I wanted to compute confidence for vae and evaluator outputs, '
            'setting the confidence weight to 1. immediately pushes the confidence to 1.0.'
        )

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args()
        self.opt.is_train = self.is_train  # train or test
        if self.opt.is_train:
            self.opt.dataset_split = "train"
        else:
            self.opt.dataset_split = "test"
        self.opt.batch_size = self.opt.num_objects_per_batch * \
            self.opt.num_grasps_per_object
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

        if self.is_train:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # save to the disk
            name = self.opt.arch
            name += "_lr_" + str(self.opt.lr).split(".")[-1] + "_bs_" + str(
                self.opt.batch_size)
            name += "_scale_" + str(self.opt.model_scale) + "_npoints_" + str(
                self.opt.pointnet_nclusters) + "_radius_" + str(
                    self.opt.pointnet_radius).split(".")[-1]
            if self.opt.arch == "vae" or self.opt.arch == "gan":
                name += "_latent_size_" + str(self.opt.latent_size)

            self.opt.name = name
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            if os.path.isdir(expr_dir) and not self.opt.continue_train:
                option = "Directory " + expr_dir + \
                    " already exists and you have not chosen to continue to train.\nDo you want to override that training instance with a new one the press (Y/N)."
                print(option)
                while True:
                    choice = input()
                    if choice.upper() == "Y":
                        print("Overriding directory " + expr_dir)
                        shutil.rmtree(expr_dir)
                        utils.mkdir(expr_dir)
                        break
                    elif choice.upper() == "N":
                        print(
                            "Terminating. Remember, if you want to continue to train from a saved instance then run the script with the flag --continue_train"
                        )
                        return None
            else:
                utils.mkdir(expr_dir)

            yaml_path = os.path.join(expr_dir, 'opt.yaml')
            with open(yaml_path, 'w') as yaml_file:
                yaml.dump(args, yaml_file)

            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
