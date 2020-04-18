import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import losses
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import pointnet2_ops.pointnet2_modules as pointnet2


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count -
                             opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type, init_gain, gpu_ids):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net


def define_classifier(opt, gpu_ids, arch, init_type, init_gain):
    net = None
    if arch == 'vae':
        net = GraspSamplerVAE(opt.model_scale, opt.pointnet_radius,
                              opt.pointnet_nclusters, opt.latent_size)
    elif arch == 'gan':
        net = GraspSamplerGAN(opt.model_scale, opt.pointnet_radius,
                              opt.pointnet_nclusters, opt.latent_size)
    elif arch == 'evaluator':
        net = GraspEvaluator(opt.model_scale, opt.pointnet_radius,
                             opt.pointnet_nclusters)
    else:
        raise NotImplementedError('model name [%s] is not recognized' % arch)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_loss(opt):
    if opt.arch == 'vae':
        kl_loss = losses.kl_divergence
        reconstruction_loss = losses.control_point_l1_loss
        return kl_loss, reconstruction_loss
    elif opt.arch == 'gan':
        reconstruction_loss = losses.min_distance_loss
        return reconstruction_loss
    elif opt.arch == 'evaluator':
        loss = losses.classification_with_confidence_loss
        return loss
    else:
        raise NotImplementedError("Loss not found")


class GraspSampler(nn.Module):
    def __init__(self):
        super(GraspSampler, self).__init__()

    def decode(self, pc, z):
        pc_features = self.concatenate_z_with_pc(pc, z).transpose(-1, 1)
        z = self.decoder(pc, pc_features)
        predicted_qt = torch.cat(
            (self.q(z), F.normalize(self.t(z), p=2, dim=-1)), -1)

        return predicted_qt, F.sigmoid(self.confidence(z))

    def concatenate_z_with_pc(self, pc, z):
        z.unsqueeze_(1)
        z = z.expand(z.shape[0], pc.shape[1], z.shape[-1])
        return torch.cat((pc, z), -1)

    def create_decoder(self, model_scale, pointnet_radius, pointnet_nclusters):
        self.decoder = base_network(pointnet_radius, model_scale,
                                    pointnet_nclusters)
        self.q = nn.Linear(model_scale * 1024, 4)
        self.t = nn.Linear(model_scale * 1024, 3)
        self.confidence = nn.Linear(model_scale * 1024, 1)


class GraspSamplerVAE(GraspSampler):
    """Network for learning a generative VAE grasp-sampler
    """
    def __init__(self,
                 model_scale,
                 pointnet_radius=0.02,
                 pointnet_nclusters=128,
                 latent_size=2):
        super(GraspSamplerVAE, self).__init__()
        self.create_encoder(model_scale, pointnet_radius, pointnet_nclusters)
        self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters)
        self.create_bottleneck(model_scale * 1024, latent_size)
        self.latent_size = latent_size

    def create_encoder(
            self,
            model_scale,
            pointnet_radius,
            pointnet_nclusters,
    ):
        self.encoder = base_network(pointnet_radius, model_scale,
                                    pointnet_nclusters)

    def create_bottleneck(self, input_size, latent_size):
        mu = nn.Linear(input_size, latent_size)
        logvar = nn.Linear(input_size, latent_size)
        self.latent_space = (mu, logvar)

    def encode(self, pc, pc_features):
        return self.encoder(pc, pc_features)

    def bottleneck(self, z):
        return self.latent_space[0](z), self.latent_space[1](z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, pc, grasp=None, train=True):
        if train:
            self.forward_train(pc, grasp)
        else:
            self.forward_test(pc)

    def forward_train(self, pc, grasp):
        input_features = torch.cat((pc, grasp), 0).transpose(-1, 1)
        z = self.encode(pc, input_features)
        mu, logvar = self.bottleneck(z)
        z = self.reparameterize(mu, logvar)
        return self.decode(pc, z), mu, logvar

    def forward_test(self, pc):
        z = torch.randn((pc.shape[0], self.latent_size))
        return self.decode(pc, z)


class GraspSamplerGAN(GraspSampler):
    def __init__(self,
                 model_scale,
                 pointnet_radius,
                 pointnet_nclusters,
                 latent_size=2):
        super(GraspSamplerGAN, self).__init__()
        self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters)
        self.latent_size = latent_size

    def sample_latent(self, batch_size):
        return torch.rand(batch_size, self.latent_size)

    def forward(self, pc, grasps=None):
        z = self.sample_latent(pc.shape[0])
        return self.decode(pc, z)


class GraspEvaluator(nn.Module):
    def __init__(self,
                 model_scale=1,
                 pointnet_radius=0.02,
                 pointnet_nclusters=128):
        super(GraspEvaluator, self).__init__()
        self.create_evaluator(pointnet_radius, model_scale, pointnet_nclusters)

    def create_evaluator(self, pointnet_radius, model_scale,
                         pointnet_nclusters):
        self.evaluator = base_network(pointnet_radius, model_scale,
                                      pointnet_nclusters)
        self.predictions_logits = nn.Linear(1024 * model_scale, 2)
        self.confidence = nn.Linear(1024 * model_scale, 1)

    def forward_train(self, pc, gripper_pc):
        pc, pc_features = self.merge_pc_and_gripper_pc(pc, gripper_pc)
        x = self.evaluator(pc, pc_features)
        return F.softmax(self.predictions_logits(x)), F.sigmoid(
            self.confidence(x))

    def forward_test(self, pc, gripper_pc):
        return self.forward_train(pc, gripper_pc)

    def merge_pc_and_gripper_pc(self, pc, gripper_pc):
        """
        Merges the object point cloud and gripper point cloud and
        adds a binary auxiliary feature that indicates whether each point
        belongs to the object or to the gripper.
        """
        pc_shape = pc.shape
        gripper_shape = gripper_pc.shape
        assert (len(pc_shape) == 3)
        assert (len(gripper_shape) == 3)
        assert (pc_shape[0] == gripper_shape[0])

        npoints = pc_shape[1]
        batch_size = pc_shape[0]

        l0_xyz = torch.cat((pc, gripper_pc), 1)
        labels = [
            torch.ones(pc.shape[1], 1, dtype=torch.float32),
            torch.zeros(gripper_pc.shape[1], 1, dtype=torch.float32)
        ]
        labels = torch.cat(labels, 0)
        labels.unsqueeze_(0)
        labels = torch.tile(labels, [batch_size, 1, 1])

        l0_points = torch.cat([l0_xyz, labels], -1).transpose(-1, 1)

        return l0_xyz, l0_points


def base_network(pointnet_radius, pointnet_nclusters, scale):
    sa1_module = pointnet2.PointnetSAModule(
        npoints=pointnet_nclusters,
        radius=pointnet_radius,
        nsamples=64,
        mlp=[64 * scale, 64 * scale, 128 * scale])
    sa2_module = pointnet2.PointnetSAModule(
        npoints=32,
        radius=0.04,
        nsamples=128,
        mlp=[128 * scale, 128 * scale, 256 * scale])
    sa3_module = pointnet2.PointnetSAModule(
        mlp=[256 * scale, 256 * scale, 512 * scale])
    fc_layer = nn.Sequential(nn.Linear(1024, 1024 * scale),
                             nn.BatchNorm1d(1024 * scale), nn.ReLU(True),
                             nn.Linear(1024 * scale, 1024 * scale),
                             nn.BatchNorm1d(1024 * scale), nn.ReLU(True))
    return nn.ModuleList([sa1_module, sa2_module, sa3_module, fc_layer])