from torch.utils.data import DataLoader
from torchvision import transforms
from utils import transform as T
from datasets import get_dataset

Dataset = get_dataset('water')

class Transformation(object):
    def __init__(self, scales, cropsize):
        self.trans_func = T.Compose([
            T.RandomResizedCrop(scales, cropsize),
            T.RandomHorizontalFlip(),
            T.ColorJitter(
                brightness = 0.4,
                contrast = 0.4,
                saturation = 0.4
            ),
        ]) 

    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)
        return im_lb

# Configure your dataloader here.
# And you can call or define another dataset outside to bring it here.
def getDataLoader(cfg, dataset = None, mode = 'train'):
    dataset = Dataset(
        cfg,
        mode = mode,
        transform = Transformation(cfg.transform.scales, cfg.transform.cropsize)
    ) if not dataset else dataset

    return DataLoader(
        dataset,
        batch_size = cfg.dataloader.batch_size,
        shuffle = cfg.dataloader.isShuffle,
        num_workers = cfg.dataloader.num_worker
    )