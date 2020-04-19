import numpy as np
import copy
import os
import math
import time
import trimesh.transformations as tra
import json
from utils import sample
import torch

GRIPPER_PC = np.load('gripper_models/panda_pc.npy',
                     allow_pickle=True).item()['points']
GRIPPER_PC[:, 3] = 1.


def farthest_points(data,
                    nclusters,
                    dist_func,
                    return_center_indexes=False,
                    return_distances=False,
                    verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of 
          clusters.
        return_distances: bool, If True, return distances of each point from centers.
      
      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in 
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of 
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0],
                             dtype=np.int32), np.arange(data.shape[0],
                                                        dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0], ), dtype=np.int32) * -1
    distances = np.ones((data.shape[0], ), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(
                np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def distance_by_translation_grasp(p1, p2):
    """
      Gets two nx4x4 numpy arrays and computes the translation of all the
      grasps.
    """
    t1 = p1[:, :3, 3]
    t2 = p2[:, :3, 3]
    return np.sqrt(np.sum(np.square(t1 - t2), axis=-1))


def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates whether to use farthest point sampling
      to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc,
                                                npoints,
                                                distance_by_translation_point,
                                                return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]),
                                              size=npoints,
                                              replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.catenate((pc, pc[index, :]), axis=0)
    return pc


def perturb_grasp(grasp, num, min_translation, max_translation, min_rotation,
                  max_rotation):
    """
      Self explanatory.
    """
    output_grasps = []
    for _ in range(num):
        sampled_translation = [
            np.random.uniform(lb, ub)
            for lb, ub in zip(min_translation, max_translation)
        ]
        sampled_rotation = [
            np.random.uniform(lb, ub)
            for lb, ub in zip(min_rotation, max_rotation)
        ]
        grasp_transformation = tra.euler_matrix(*sampled_rotation)
        grasp_transformation[:3, 3] = sampled_translation
        output_grasps.append(np.matmul(grasp, grasp_transformation))

    return output_grasps


def evaluate_grasps(grasp_tfs, obj_mesh):
    """
        Check the collision of the grasps and also heuristic quality for each
        grasp.
    """
    collisions, _ = sample.in_collision_with_gripper(
        obj_mesh,
        grasp_tfs,
        gripper_name='panda',
        silent=True,
    )
    qualities = sample.grasp_quality_point_contacts(
        grasp_tfs,
        collisions,
        object_mesh=obj_mesh,
        gripper_name='panda',
        silent=True,
    )

    return np.asarray(collisions), np.asarray(qualities)


def inverse_transform(trans):
    """
      Computes the inverse of 4x4 transform.
    """
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t

    return output


def uniform_quaternions():
    quaternions = [
        l[:-1].split('\t') for l in open(
            '../uniform_quaternions/data2_4608.qua', 'r').readlines()
    ]

    quaternions = [[float(t[0]),
                    float(t[1]),
                    float(t[2]),
                    float(t[3])] for t in quaternions]
    quaternions = np.asarray(quaternions)
    quaternions = np.roll(quaternions, 1, axis=1)
    return [tra.quaternion_matrix(q) for q in quaternions]


def nonuniform_quaternions():
    all_poses = []
    for az in np.linspace(0, np.pi * 2, 30):
        for el in np.linspace(-np.pi / 2, np.pi / 2, 30):
            all_poses.append(tra.euler_matrix(el, az, 0))
    return all_poses


def print_network(net):
    """Print the total number of parameters in the network
    Parameters:
        network
    """
    print('---------- Network initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')


def merge_pc_and_gripper_pc(pc,
                            gripper_pc,
                            instance_mode=0,
                            pc_latent=None,
                            gripper_pc_latent=None):
    """
    Merges the object point cloud and gripper point cloud and
    adds a binary auxilary feature that indicates whether each point
    belongs to the object or to the gripper.
    """

    pc_shape = pc.shape
    gripper_shape = gripper_pc.shape
    assert (len(pc_shape) == 3)
    assert (len(gripper_shape) == 3)
    assert (pc_shape[0] == gripper_shape[0])

    npoints = pc.shape[1]
    batch_size = pc.shape[0]

    if instance_mode == 1:
        assert pc_shape[-1] == 3
        latent_dist = [pc_latent, gripper_pc_latent]
        latent_dist = torch.cat(latent_dist, 1)

    l0_xyz = torch.cat((pc, gripper_pc), 1)
    labels = [
        torch.ones((pc.shape[1], 1), dtype=torch.float32),
        torch.zeros((gripper_pc.shape[1], 1), dtype=torch.float32)
    ]
    labels = torch.cat(labels, 0)
    labels = torch.expand_dims(labels, 0)
    labels = torch.tile(labels, [batch_size, 1, 1])

    if instance_mode == 1:
        l0_points = torch.cat([l0_xyz, latent_dist, labels], -1)
    else:
        l0_points = torch.cat([l0_xyz, labels], -1)

    return l0_xyz, l0_points


def get_gripper_pc(batch_size, npoints, use_torch=True):
    """
      Returns a numpy array or a tensor of shape (batch_size x npoints x 4).
      Represents gripper with the sepcified number of points.
      use_tf: switches between output tensor or numpy array.
    """
    output = np.copy(GRIPPER_PC)
    if npoints != -1:
        assert (npoints > 0 and npoints <= output.shape[0]
                ), 'gripper_pc_npoint is too large {} > {}'.format(
                    npoints, output.shape[0])
        output = output[:npoints]
        output = np.expand_dims(output, 0)
    else:
        raise ValueError('npoints should not be -1.')

    if use_torch:
        output = torch.tensor(output, torch.float32)
        output = output.repeat(batch, size, 1, 1)
        return output
    else:
        output = np.tile(output, [batch_size, 1, 1])

    return output


def get_control_point_tensor(batch_size, use_torch=True):
    """
      Outputs a tensor of shape (batch_size x 6 x 3).
      use_tf: switches between outputing a tensor and outputing a numpy array.
    """
    control_points = np.load('./gripper_control_points/panda.npy')[:, :3]
    control_points = [[0, 0, 0], [0, 0, 0], control_points[0, :],
                      control_points[1, :], control_points[-2, :],
                      control_points[-1, :]]
    control_points = np.asarray(control_points, dtype=np.float32)
    control_points = np.tile(np.expand_dims(control_points, 0),
                             [batch_size, 1, 1])

    if use_torch:
        return torch.tensor(control_points)

    return control_points


def transform_control_points(gt_grasps,
                             batch_size,
                             mode='qt',
                             scope='transform_gt_control_points'):
    """
      Transforms canonical points using gt_grasps.
      mode = 'qt' expects gt_grasps to have (batch_size x 7) where each 
        element is catenation of quaternion and translation for each
        grasps.
      mode = 'rt': expects to have shape (batch_size x 4 x 4) where
        each element is 4x4 transformation matrix of each grasp.
    """
    assert (mode == 'qt' or mode == 'rt'), mode
    grasp_shape = gt_grasps.shape
    if mode == 'qt':
        assert (len(grasp_shape) == 2), grasp_shape
        assert (grasp_shape[-1] == 7), grasp_shape
        control_points = get_control_point_tensor(batch_size).cuda()
        num_control_points = control_points.shape[1]
        input_gt_grasps = gt_grasps
        gt_grasps = torch.unsqueeze(input_gt_grasps,
                                    1).repeat(1, num_control_points, 1)
        gt_q = gt_grasps[:, :, :4]
        gt_t = gt_grasps[:, :, 4:]
        gt_control_points = rotate_point_by_quaternion(control_points, gt_q)
        gt_control_points += gt_t

        return gt_control_points
    else:
        assert (len(grasp_shape) == 3), grasp_shape
        assert (grasp_shape[1] == 4 and grasp_shape[2] == 4), grasp_shape
        control_points = get_control_point_tensor(batch_size)
        shape = control_points.shape
        ones = torch.ones((shape[0], shape[1], 1), dtype=torch.float32)
        control_points = torch.cat((control_points, ones), -1)
        return torch.matmul(control_points, gt_grasps.permute(2, 0, 1))


def transform_control_points_numpy(gt_grasps,
                                   batch_size,
                                   mode='qt',
                                   scope='transform_gt_control_points'):
    """
      Transforms canonical points using gt_grasps.
      mode = 'qt' expects gt_grasps to have (batch_size x 7) where each 
        element is catenation of quaternion and translation for each
        grasps.
      mode = 'rt': expects to have shape (batch_size x 4 x 4) where
        each element is 4x4 transformation matrix of each grasp.
    """
    assert (mode == 'qt' or mode == 'rt'), mode
    grasp_shape = gt_grasps.shape
    if mode == 'qt':
        assert (len(grasp_shape) == 2), grasp_shape
        assert (grasp_shape[-1] == 7), grasp_shape
        control_points = get_control_point_tensor(batch_size, use_torch=False)
        num_control_points = control_points.shape[1]
        input_gt_grasps = gt_grasps
        gt_grasps = np.expand_dims(input_gt_grasps,
                                   1).repeat(num_control_points, axis=1)
        gt_q = gt_grasps[:, :, :4]
        gt_t = gt_grasps[:, :, 4:]

        gt_control_points = rotate_point_by_quaternion(control_points, gt_q)
        gt_control_points += gt_t

        return gt_control_points
    else:
        assert (len(grasp_shape) == 3), grasp_shape
        assert (grasp_shape[1] == 4 and grasp_shape[2] == 4), grasp_shape
        control_points = get_control_point_tensor(batch_size, use_torch=False)
        shape = control_points.shape
        ones = np.ones((shape[0], shape[1], 1), dtype=np.float32)
        control_points = np.concatenate((control_points, ones), -1)
        return np.einsum("ijk,kki->ijk", control_points, gt_grasps.T)


def quaternion_mult(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def conj_quaternion(q):
    """
      Conjugate of quaternion q.
    """
    q_conj = q.clone()
    q_conj[:, :, 1:] *= -1
    return q_conj


def rotate_point_by_quaternion(point, q):
    """
      Takes in points with shape of (batch_size x n x 3) and quaternions with
      shape of (batch_size x n x 4) and returns a tensor with shape of 
      (batch_size x n x 3) which is the rotation of the point with quaternion
      q. 
    """
    shape = point.shape
    q_shape = q.shape

    assert (len(shape) == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (shape[-1] == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (len(q_shape) == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (q_shape[-1] == 4), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (q_shape[1] == shape[1]), 'point shape = {} q shape = {}'.format(
        shape, q_shape)

    q_conj = conj_quaternion(q)
    r = torch.cat([
        torch.zeros((shape[0], shape[1], 1), dtype=point.dtype).cuda(), point
    ],
                  dim=-1)
    final_point = quaternion_mult(quaternion_mult(q, r), q_conj)
    final_output = final_point[:, :,
                               1:]  #torch.slice(final_point, [0, 0, 1], shape)
    return final_output


def tc_rotation_matrix(az, el, th, batched=False):
    if batched:
        cx = torch.cos(torch.reshape(az, [-1, 1]))
        cy = torch.cos(torch.reshape(el, [-1, 1]))
        cz = torch.cos(torch.reshape(th, [-1, 1]))
        sx = torch.sin(torch.reshape(az, [-1, 1]))
        sy = torch.sin(torch.reshape(el, [-1, 1]))
        sz = torch.sin(torch.reshape(th, [-1, 1]))

        ones = torch.ones_like(cx)
        zeros = torch.zeros_like(cx)

        rx = torch.cat([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx],
                       dim=-1)
        ry = torch.cat([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy],
                       dim=-1)
        rz = torch.cat([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones],
                       dim=-1)

        rx = torch.reshape(rx, [-1, 3, 3])
        ry = torch.reshape(ry, [-1, 3, 3])
        rz = torch.reshape(rz, [-1, 3, 3])

        return torch.matmul(rz, torch.matmul(ry, rx))
    else:
        cx = torch.cos(az)
        cy = torch.cos(el)
        cz = torch.cos(th)
        sx = torch.sin(az)
        sy = torch.sin(el)
        sz = torch.sin(th)

        rx = torch.stack([[1., 0., 0.], [0, cx, -sx], [0, sx, cx]], dim=0)
        ry = torch.stack([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dim=0)
        rz = torch.stack([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dim=0)

        return torch.matmul(rz, torch.matmul(ry, rx))


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
