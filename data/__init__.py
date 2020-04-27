import torch.utils.data
from data.base_dataset import collate_fn
import threading


def CreateDataset(opt):
    """loads dataset class"""

    if opt.arch == 'vae' or opt.arch == 'gan':
        from data.grasp_sampling_data import GraspSamplingData
        dataset = GraspSamplingData(opt)
    else:
        from data.grasp_evaluator_data import GraspEvaluatorData
        dataset = GraspEvaluatorData(opt)
    return dataset


class DataLoader:
    """multi-threaded data loading"""
    def __init__(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.num_objects_per_batch,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            collate_fn=collate_fn)

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
