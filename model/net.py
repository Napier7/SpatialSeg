import torch
import torch.nn as nn

'''
@description: Decoder block (only including a 3×3 conv layer)
@param {number} in_channel: input channel
@param {number} out_channel: output channel 
'''
class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super(Decoder, self).__init__()
        self.conv = nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size = 3,
                stride = 2,
                padding = 1
        )
        # self.relu = nn.ReLU(inplace=True)
        # self.init_weight()
    
    # forward
    def forward(self, x):
        x = self.conv(x)
        # self.relu(x)
        return x
    
    # init model weight
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

        
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
class Net(nn.Module):
    def __init__(self, n_class, mode = 'train') -> None:
        super(Net, self).__init__()
        self.num = n_class
        self.mode = mode
        self.down1 = Decoder(3, 16)
        self.down2 = Decoder(16, 32)
        self.down3 = Decoder(32, 64)
        # self.up = UpSample(16, self.num, up_factor = 2)
        self.up = UpSample(64, self.num, up_factor = 8)
        self.init_weight()
    
    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
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
    net = Net(2)
    x = torch.randn(1, 3, 512, 512)
    x = net(x)
    print(x.shape)