# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function

import numpy as np
import argparse
import grasp_estimator
import sys
import os
import glob
import mayavi.mlab as mlab
from visualization_utils import *
import mayavi.mlab as mlab
from utils import utils


def make_parser():
    parser = argparse.ArgumentParser(
        description='6-DoF GraspNet Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--grasp_sampler_folder',
        type=str,
        default=
        'checkpoints/gan_lr_0002_bs_10_scale_1_npoints_128_radius_02_latent_size_2/'
    )
    parser.add_argument(
        '--grasp_evaluator_folder',
        type=str,
        default=
        'checkpoints/evaluator_lr_0002_bs_10_scale_1_npoints_128_radius_02/')
    parser.add_argument('--refinement',
                        choices={"gradient", "sampling"},
                        default='gradient')
    parser.add_argument('--refine_steps', type=int, default=10)

    parser.add_argument('--npy_folder', type=str, default='demo/data/')
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--target_pc_size', type=int, default=1024)
    parser.add_argument('--num_grasp_samples', type=int, default=10)
    parser.add_argument('--gpu', action='store_true')

    return parser


def get_color_for_pc(pc, K, color_image):
    proj = pc.dot(K.T)
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]

    pc_colors = np.zeros((pc.shape[0], 3), dtype=np.uint8)
    for i, p in enumerate(proj):
        x = int(p[0])
        y = int(p[1])
        pc_colors[i, :] = color_image[y, x, :]

    return pc_colors


def backproject(depth_cv,
                intrinsic_matrix,
                return_finite_depth=True,
                return_selection=False):

    depth = depth_cv.astype(np.float32, copy=True)

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    X = np.array(X).transpose()
    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    if return_selection:
        return X, selection

    return X


def main(args):
    parser = make_parser()
    args = parser.parse_args()
    grasp_sampler_args = utils.read_checkpoint_args(args.grasp_sampler_folder)
    grasp_sampler_args.is_train = False
    grasp_evaluator_args = utils.read_checkpoint_args(
        args.grasp_evaluator_folder)
    grasp_evaluator_args.continue_train = True
    estimator = grasp_estimator.GraspEstimator(grasp_sampler_args,
                                               grasp_evaluator_args, args)
    for npy_file in glob.glob(os.path.join(args.npy_folder, '*.npy')):
        # Depending on your numpy version you may need to change allow_pickle
        # from True to False.

        data = np.load(npy_file, allow_pickle=True, encoding="latin1").item()

        depth = data['depth']
        image = data['image']
        K = data['intrinsics_matrix']
        # Removing points that are farther than 1 meter or missing depth
        # values.
        #depth[depth == 0 or depth > 1] = np.nan

        np.nan_to_num(depth, copy=False)
        mask = np.where(np.logical_or(depth == 0, depth > 1))
        depth[mask] = np.nan
        pc, selection = backproject(depth,
                                    K,
                                    return_finite_depth=True,
                                    return_selection=True)
        pc_colors = image.copy()
        pc_colors = np.reshape(pc_colors, [-1, 3])
        pc_colors = pc_colors[selection, :]

        # Smoothed pc comes from averaging the depth for 10 frames and removing
        # the pixels with jittery depth between those 10 frames.
        object_pc = data['smoothed_object_pc']
        generated_grasps, generated_scores = estimator.generate_and_refine_grasps(
            object_pc, )
        mlab.figure(bgcolor=(1, 1, 1))
        draw_scene(
            pc,
            pc_color=pc_colors,
            grasps=generated_grasps,
            grasp_scores=generated_scores,
        )
        print('close the window to continue to next object . . .')
        mlab.show()


if __name__ == '__main__':
    main(sys.argv[1:])
