'''
Description: 
Author: 
LastEditors: Napier
LastEditTime: 2022-04-25 22:32:17
'''
import torch
import torchvision.models as Model
import torch.nn as nn
# from torchsummary import summary
# from torchstat import stat

class Resnet34(nn.Module):
     def __init__(self):
        super(Resnet34, self).__init__()
        resnet = Model.resnet34(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

     def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2 64
        x = self.maxpool(x)

        x = self.layer1(x) # 1/4 64
        x = self.layer2(x) # 1/8 128
        x = self.layer3(x) # 1/16 256
        x = self.layer4(x) # 1/32 512
    
        return x

if __name__ == '__main__':
   #  input = torch.rand(2,3,1024,1024)
    net = Resnet34()
   #  summary(net.cuda(),(3,1024,1024))
   #  stat(net, (3,1024,1024))
   #  out = net(input)
   #  print(out[1].size())