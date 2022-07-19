from torch.utils.data import DataLoader
from torchvision import transforms
from utils import transform as T
from datasets import get_dataset

Dataset = get_dataset('general')


class Transformation(object):
    def __init__(self, cfg_trans):
        # when train
        self.trans_func = T.Compose([
            T.RandomResizedCrop(cfg_trans.scales, cfg_trans.cropsize),
            T.RandomHorizontalFlip(),
            # T.ColorJitter(
            #     brightness = 0.4,
            #     contrast = 0.4,
            #     saturation = 0.4
            # ),
            T.PhotoMetricDistortion(),
            T.ToTensor(cfg_trans.mean, cfg_trans.std)
        ]) 

        # when val
        self.val_func = T.Compose([
            T.ToTensor(cfg_trans.mean, cfg_trans.std)
        ]) 

    def __call__(self, im_lb, mode = 'train'):
        im_lb = self.trans_func(im_lb) if mode == 'train' else self.val_func(im_lb) 
        return im_lb

# Configure your dataloader here.
# And you can call or define another dataset outside to bring it here.
def getDataLoader(cfg, dataset = None, mode = 'train'):
    dataset = Dataset(
        cfg,
        mode = mode,
        transform = Transformation(cfg.transform)
    ) if not dataset else dataset

    return DataLoader(
        dataset,
        batch_size = cfg.dataloader.batch_size,
        shuffle = cfg.dataloader.isShuffle,
        num_workers = cfg.dataloader.num_worker
    )