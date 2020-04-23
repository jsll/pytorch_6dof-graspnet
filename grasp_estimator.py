from __future__ import print_function

from models import create_model
import numpy as np
import torch
import time
import trimesh
import trimesh.transformations as tra
#import surface_normal
import copy
import os
from utils import utils


class GraspEstimator:
    """
      Includes the code used for running the inference.
    """
    def __init__(self, grasp_sampler_opt, grasp_evaluator_opt, opt):
        self.grasp_sampler_opt = grasp_sampler_opt
        self.grasp_evaluator_opt = grasp_evaluator_opt
        self.opt = opt
        self.target_pc_size = opt.target_pc_size
        self.num_refine_steps = opt.refine_steps
        self.refine_method = opt.refinement_method
        self.num_grasp_samples = opt.num_grasp_samples
        self.threshold = opt.threshold
        self.use_gpu = True  # opt.use_gpu
        self.choose_fn = None
        self.choose_fns = {
            "all":
            None,
            "better_than_threshold":
            utils.choose_grasps_better_than_threshold,
            "better_than_threshold_in_sequence":
            utils.choose_grasps_better_than_threshold_in_sequence,
        }
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")
        self.grasp_evaluator = create_model(grasp_evaluator_opt)
        self.grasp_sampler = create_model(grasp_sampler_opt)

    def generate_and_refine_grasps(
            self,
            pc,
    ):
        pc, pc_mean = self.prepare_pc(pc)
        grasps, confidence, z = self.generate_grasps(pc)
        improved_eulers, improved_ts, improved_success = self.refine_grasps(
            pc, grasps, self.refine_method, self.num_refine_steps)
        improved_success = np.asarray(improved_success).squeeze()
        if self.choose_fn is None:
            selection_mask = np.ones(improved_success.shape, dtype=np.float32)
        else:
            selection_mask = self.choose_fns[self.choose_fn](improved_eulers,
                                                             improved_ts,
                                                             selection_mask,
                                                             self.threshold)
        grasps = utils.rot_and_trans_to_grasps(improved_eulers, improved_ts,
                                               selection_mask)
        utils.denormalize_grasps(grasps, pc_mean)
        refine_indexes, sample_indexes = np.where(selection_mask)
        success_prob = improved_success[refine_indexes, sample_indexes].tolist(
        )
        return grasps, success_prob

    def prepare_pc(self, pc):
        if pc.shape[0] > self.target_pc_size:
            pc = utils.regularize_pc_point_count(pc, self.target_pc_size)
        pc_mean = np.mean(pc, 0)
        pc -= np.expand_dims(pc_mean, 0)

        pc = np.tile(pc, (self.num_grasp_samples, 1, 1))
        pc = torch.from_numpy(pc).float().to(self.device)
        return pc, pc_mean

    def generate_grasps(self, pc):
        return self.grasp_sampler.generate_grasps(pc)

    def refine_grasps(self, pc, grasps, refine_method, num_refine_steps=10):

        grasp_eulers, grasp_translations = utils.convert_qt_to_rt(grasps)
        if refine_method == "gradient":
            improve_fun = self.improve_grasps_gradient_based
            grasp_eulers = torch.autograd.Variable(grasp_eulers.to(
                self.device),
                                                   requires_grad=True)
            grasp_translations = torch.autograd.Variable(grasp_translations.to(
                self.device),
                                                         requires_grad=True)

        else:
            improve_fun = self.improve_grasps_sampling_based

        improved_success = []
        improved_eulers = []
        improved_ts = []
        improved_eulers.append(grasp_eulers.cpu().data.numpy())
        improved_ts.append(grasp_translations.cpu().data.numpy())
        last_success = None
        for i in range(num_refine_steps):
            success_prob, last_success = improve_fun(pc, grasp_eulers,
                                                     grasp_translations,
                                                     last_success)
            improved_success.append(success_prob.cpu().data.numpy())
            improved_eulers.append(grasp_eulers.cpu().data.numpy())
            improved_ts.append(grasp_translations.cpu().data.numpy())

        # we need to run the success on the final improved grasps
        grasp_pcs = utils.control_points_from_rot_and_trans(
            grasp_eulers, grasp_translations, self.device)
        improved_success.append(
            self.grasp_evaluator.evaluate_grasps(
                pc, grasp_pcs).squeeze().cpu().data.numpy())

        return np.asarray(improved_eulers), np.asarray(
            improved_ts), np.asarray(improved_success)

    def improve_grasps_gradient_based(
            self, pcs, grasp_eulers, grasp_trans, last_success
    ):  #euler_angles, translation, eval_and_improve, metadata):
        grasp_pcs = utils.control_points_from_rot_and_trans(
            grasp_eulers, grasp_trans, self.device)

        success = self.grasp_evaluator.evaluate_grasps(pcs, grasp_pcs)
        success.squeeze().backward(
            torch.ones(success.shape[0]).to(self.device))
        delta_t = grasp_trans.grad
        norm_t = torch.norm(delta_t, p=2, dim=-1).to(self.device)
        # Adjust the alpha so that it won't update more than 1 cm. Gradient is only valid
        # in small neighborhood.
        alpha = torch.min(0.01 / norm_t, torch.tensor(1.0).to(self.device))
        grasp_trans.data += grasp_trans.grad * alpha[:, None]
        temp = grasp_eulers.clone()
        grasp_eulers.data += grasp_eulers.grad * alpha[:, None]
        return success.squeeze(), None

    def improve_grasps_sampling_based(self,
                                      pcs,
                                      grasp_eulers,
                                      grasp_trans,
                                      last_success=None):
        if last_success is None:
            grasp_pcs = utils.control_points_from_rot_and_trans(
                grasp_eulers, grasp_trans, self.device)
            last_success = self.grasp_evaluator.evaluate_grasps(pcs, grasp_pcs)

        delta_t = 2 * (torch.rand(grasp_trans.shape).to(self.device) - 0.5)
        delta_t *= 0.02
        delta_euler_angles = (torch.rand(grasp_eulers.shape).to(self.device) -
                              0.5) * 2
        perturbed_translation = grasp_trans + delta_t
        perturbed_euler_angles = grasp_eulers + delta_euler_angles
        grasp_pcs = utils.control_points_from_rot_and_trans(
            perturbed_euler_angles, perturbed_translation, self.device)

        perturbed_success = self.grasp_evaluator.evaluate_grasps(
            pcs, grasp_pcs)
        ratio = perturbed_success / torch.max(
            last_success,
            torch.tensor(0.0001).to(self.device))

        mask = torch.rand(ratio.shape).to(self.device) <= ratio

        next_success = last_success
        ind = torch.where(mask)[0]
        next_success[ind] = perturbed_success[ind]
        grasp_trans[ind].data = perturbed_translation.data[ind]
        grasp_eulers[ind].data = perturbed_euler_angles.data[ind]
        return last_success.squeeze(), next_success