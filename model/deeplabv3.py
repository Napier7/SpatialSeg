'''
Description: DeeplabV3 (resnet-18) 
Author: Napier
LastEditors: Napier
LastEditTime: 2022-04-27 16:22:16
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

# from .backbone.resnet18 import Resnet18
from .backbone.resnet34 import Resnet34
class ASPP(nn.Module):
    def __init__(self, num_classes):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 512, h/32, w/32)) 

        feature_map_h = feature_map.size()[2] # (== h/32)
        feature_map_w = feature_map.size()[3] # (== w/32)
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/32, w/32))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/32, w/32))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/32, w/32))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/32, w/32))

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
        out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes, h/16, w/16))

        return out
        
class DeepLabV3(nn.Module):
    def __init__(self, n_class = 2, mode = 'train'):
        super(DeepLabV3, self).__init__()
        self.mode = mode
        self.num_classes = n_class
        # self.resnet = Resnet18()
        self.resnet = Resnet34()
        self.aspp = ASPP(num_classes=self.num_classes)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x)[-1] # (shape: (batch_size, 512, h/32, w/32)) 
        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/32, w/32))

        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))
        if self.mode == 'train' or self.mode == 'val':
            return output
        elif self.mode == 'pred':
            output = output.argmax(dim = 1)
            return output


if __name__ == "__main__":
    net = DeepLabV3(2)
    x = torch.randn(2, 3, 1024, 1024)
    out = net(x)
    print(out.shape)
    # stat(net, (3,1024,1024))
    # summary(net.cuda(), (3,1024,1024))