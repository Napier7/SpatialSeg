import os
from torch.utils.data import Dataset
import cv2
class GeneralDataset(Dataset):
    # You can configure this dataset by cfg except transform.
    # Certainly, you can also define them by yourself in some complicated situation.
    def __init__(self, cfg, mode = 'train', items = None, im_root = None, label_root = None, transform = None):
        self.mode = mode
        self.cfg = cfg
        self.transform = transform if not transform else transform
        self.items = self.getItems(mode) if not items else items
        self.im_root = cfg.dataset.im_root if not im_root else im_root
        self.label_root = cfg.dataset.label_root if not label_root else label_root
        self.n_class = cfg.model.n_class

    # Dataloader loads data by calling this method.
    # And you can process data here before loading.
    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.im_root, self.items[index]) + '.png')
        label = cv2.imread(os.path.join(self.label_root, self.items[index]) +'.png', 0) if self.mode != 'test' else None

        # transforms
        im_lb = dict(im = img, lb = label)
        if self.transform is not None and (self.mode == 'train' or self.mode == 'val'):
            im_lb = self.transform(im_lb)
        img, label = im_lb['im'], im_lb['lb']

        return img, label

    def __len__(self):
        return len(self.items)

    # Get data items list from annotation file through seting mode.
    # It will not work in mode == test. Because most of the test sets do not have an annotation file and we didn't configgure it in cfg.
    def getItems(self, mode):
        cfg = self.cfg.dataset
        file = None
        if mode == 'train':
            file = open(cfg.train_im_anns)
        elif mode == 'val':
            file = open(cfg.val_im_anns)
        # elif mode == 'test':
        #     file = open(cfg.test_im_anns)
        else:
            print('Please check TarinLoader mode accurately. If you set mode == test, please define a items list for Dataset.')
            return
        return file.read().splitlines()