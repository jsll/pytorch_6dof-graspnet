import torch
from . import networks
from os.path import join
import utils.utils as utils


class GraspNetModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> sampling / evaluation)
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        if self.gpu_ids and self.gpu_ids[0] >= torch.cuda.device_count():
            self.gpu_ids[0] = torch.cuda.device_count() - 1
        self.device = torch.device('cuda:{}'.format(
            self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.loss = None
        self.pcs = None
        self.grasps = None
        # load/define networks
        self.net = networks.define_classifier(opt, self.gpu_ids, opt.arch,
                                              opt.init_type, opt.init_gain,
                                              self.device)

        self.criterion = networks.define_loss(opt)

        self.confidence_loss = None
        if self.opt.arch == "vae":
            self.kl_loss = None
            self.reconstruction_loss = None
        elif self.opt.arch == "gan":
            self.reconstruction_loss = None
        else:
            self.classification_loss = None

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=opt.lr,
                                              betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch, self.is_train)

    def set_input(self, data):
        input_pcs = torch.from_numpy(data['pc']).contiguous()
        input_grasps = torch.from_numpy(data['grasp_rt']).float()
        if self.opt.arch == "evaluator":
            targets = torch.from_numpy(data['labels']).float()
        else:
            targets = torch.from_numpy(data['target_cps']).float()
        self.pcs = input_pcs.to(self.device).requires_grad_(self.is_train)
        self.grasps = input_grasps.to(self.device).requires_grad_(
            self.is_train)
        self.targets = targets.to(self.device)

    def generate_grasps(self, pcs, z=None):
        with torch.no_grad():
            return self.net.module.generate_grasps(pcs, z=z)

    def evaluate_grasps(self, pcs, gripper_pcs):
        success, _ = self.net.module(pcs, gripper_pcs)
        return torch.sigmoid(success)

    def forward(self):
        return self.net(self.pcs, self.grasps, train=self.is_train)

    def backward(self, out):
        if self.opt.arch == 'vae':
            predicted_cp, confidence, mu, logvar = out
            predicted_cp = utils.transform_control_points(
                predicted_cp, predicted_cp.shape[0], device=self.device)
            self.reconstruction_loss, self.confidence_loss = self.criterion[1](
                predicted_cp,
                self.targets,
                confidence=confidence,
                confidence_weight=self.opt.confidence_weight,
                device=self.device)
            self.kl_loss = self.opt.kl_loss_weight * self.criterion[0](
                mu, logvar, device=self.device)
            self.loss = self.kl_loss + self.reconstruction_loss + self.confidence_loss
        elif self.opt.arch == 'gan':
            predicted_cp, confidence = out
            predicted_cp = utils.transform_control_points(
                predicted_cp, predicted_cp.shape[0], device=self.device)
            self.reconstruction_loss, self.confidence_loss = self.criterion(
                predicted_cp,
                self.targets,
                confidence=confidence,
                confidence_weight=self.opt.confidence_weight,
                device=self.device)
            self.loss = self.reconstruction_loss + self.confidence_loss
        elif self.opt.arch == 'evaluator':
            grasp_classification, confidence = out
            self.classification_loss, self.confidence_loss = self.criterion(
                grasp_classification.squeeze(),
                self.targets,
                confidence,
                self.opt.confidence_weight,
                device=self.device)
            self.loss = self.classification_loss + self.confidence_loss

        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()


##################

    def load_network(self, which_epoch, train=True):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        checkpoint = torch.load(load_path, map_location=self.device)
        if hasattr(checkpoint['model_state_dict'], '_metadata'):
            del checkpoint['model_state_dict']._metadata
        net.load_state_dict(checkpoint['model_state_dict'])
        if train:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.opt.epoch_count = checkpoint["epoch"]
        else:
            net.eval()

    def save_network(self, net_name, epoch_num):
        """save model to disk"""
        save_filename = '%s_net.pth' % (net_name)
        save_path = join(self.save_dir, save_filename)
        torch.save(
            {
                'epoch': epoch_num + 1,
                'model_state_dict': self.net.module.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, save_path)

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.net.cuda(self.gpu_ids[0])

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            prediction, confidence = out
            if self.opt.arch == "vae":
                predicted_cp = utils.transform_control_points(
                    prediction, prediction.shape[0], device=self.device)
                reconstruction_loss, _ = self.criterion[1](
                    predicted_cp,
                    self.targets,
                    confidence=confidence,
                    confidence_weight=self.opt.confidence_weight,
                    device=self.device)
                return reconstruction_loss, 1
            elif self.opt.arch == "gan":
                predicted_cp = utils.transform_control_points(
                    prediction, prediction.shape[0], device=self.device)
                reconstruction_loss, _ = self.criterion(
                    predicted_cp,
                    self.targets,
                    confidence=confidence,
                    confidence_weight=self.opt.confidence_weight,
                    device=self.device)
                return reconstruction_loss, 1
            else:

                predicted = torch.round(torch.sigmoid(prediction)).squeeze()
                correct = (predicted == self.targets).sum().item()
                return correct, len(self.targets)
