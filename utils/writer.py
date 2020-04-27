import os
import time
import numpy as np
try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print('tensorboard X not installed, visualizing wont be available')
    SummaryWriter = None


class Writer:
    def __init__(self, opt):
        self.name = opt.name
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.log_name = os.path.join(self.save_dir, 'loss_log.txt')
        self.testacc_log = os.path.join(self.save_dir, 'testacc_log.txt')
        self.start_logs()
        self.nexamples = 0
        self.confidence_acc = 0
        self.ncorrect = 0

        if opt.is_train and not opt.no_vis and SummaryWriter is not None:
            self.display = SummaryWriter(
                logdir=os.path.join(self.opt.checkpoints_dir, self.opt.name) +
                "/tensorboard")  #comment=opt.name)
        else:
            self.display = None

    def start_logs(self):
        """ creates test / train log files """
        if self.opt.is_train:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write(
                    '================ Training Loss (%s) ================\n' %
                    now)
        else:
            with open(self.testacc_log, "a") as log_file:
                now = time.strftime("%c")
                log_file.write(
                    '================ Testing Acc (%s) ================\n' %
                    now)

    def print_current_losses(self,
                             epoch,
                             i,
                             losses,
                             t,
                             t_data,
                             loss_types="total_loss"):
        """ prints train loss to terminal / file """
        if type(losses) == list:
            message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f)' \
                    % (epoch, i, t, t_data)
            for (loss_type, loss_value) in zip(loss_types, losses):
                message += ' %s: %.3f' % (loss_type, loss_value.item())
        else:
            message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) loss: %.3f ' \
                    % (epoch, i, t, t_data, losses.item())
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_loss(self, losses, epoch, i, n, loss_types):
        iters = i + (epoch - 1) * n
        if self.display:
            if type(losses) == list:
                for (loss_type, loss_value) in zip(loss_types, losses):
                    self.display.add_scalar('data/train_loss/' + loss_type,
                                            loss_value, iters)
            else:
                self.display.add_scalar('data/train_loss', losses, iters)

    def plot_model_wts(self, model, epoch):
        if self.opt.is_train and self.display:
            for name, param in model.net.named_parameters():
                self.display.add_histogram(name,
                                           param.clone().cpu().data.numpy(),
                                           epoch)

    def print_acc(self, epoch, acc):
        """ prints test accuracy to terminal / file """
        if self.opt.arch == "evaluator":
            message = 'epoch: {}, TEST ACC: [{:.5} %]\n' \
                .format(epoch, acc * 100)
        else:
            message = 'epoch: {}, TEST REC LOSS: [{:.5}]\n' \
                .format(epoch, acc)

        print(message)
        with open(self.testacc_log, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_acc(self, acc, epoch):
        if self.display:
            if self.opt.arch == "evaluator":
                self.display.add_scalar('data/test_acc/grasp_prediction', acc,
                                        epoch)
            else:
                self.display.add_scalar('data/test_loss/grasp_reconstruction',
                                        acc, epoch)

    def reset_counter(self):
        """
        counts # of correct examples
        """
        self.ncorrect = 0
        self.nexamples = 0

    def update_counter(self, ncorrect, nexamples):
        self.nexamples += nexamples
        self.ncorrect += ncorrect

    @property
    def acc(self):
        return float(self.ncorrect) / self.nexamples

    def close(self):
        if self.display is not None:
            self.display.close()
