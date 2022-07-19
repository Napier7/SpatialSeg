import torch
import torch.nn as nn
from .backbone.resnet18 import Resnet18
from .backbone.resnet34 import Resnet34
        
'''
@description: upsample block (including a 1×1 conv layer and a bilinear interpolation layer)
@param {number} in_channel: input channel 
@param {number} out_channel: output channel
@param {number} up_factor:  interpolation rate
'''
class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel, up_factor) -> None:
        super(UpSample, self).__init__()
        self.conv = nn.Conv2d(in_channel,
            out_channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.up =nn.Upsample(scale_factor=up_factor,
            mode = 'bilinear',
            align_corners=False)
        self.init_weight()

    # forward
    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x
    
    # init model weight
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    


'''
@description: a easy model
@param {number} n_class: number of output class 
@param {string} mode: train/val/pred mode  
'''
class Resnet(nn.Module):
    def __init__(self, n_class, mode = 'train') -> None:
        super(Resnet, self).__init__()
        self.num = n_class
        self.mode = mode
        # self.resnet = Resnet18()
        self.resnet = Resnet34()
        self.up = UpSample(512, self.num, up_factor = 32)
        self.init_weight()
    
    def forward(self, x):
        x = self.resnet(x)[-1]
        x = self.up(x)
        if self.mode == 'train' or self.mode == 'val':
            return x
        elif self.mode == 'pred':
            return x.argmax(dim = 1)
    
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

        
if __name__ == '__main__':
    net = Resnet(2)
    x = torch.randn(1, 3, 1024, 1024)
    x = net(x)
    print(x.shape)