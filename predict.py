import torch
from model import get_model
import cv2
import numpy as np
from utils import transform as T


# 模型实例
Resnet = get_model('resnet')
resnet = Resnet(2, mode = 'pred')
resnet.eval()

# 参数载入
checkpoint = torch.load(r'../autodl-tmp/checkpoints/water/resnet/20220609/Epoch4-Train_Loss0.0175-Val_Loss0.0153.pth')
resnet.load_state_dict(checkpoint['net'])

# 数据加载
img = cv2.imread(r'../autodl-tmp/water-samples/total_sample/img/H48F017018_clip3_12.png')
trans = T.ToTensor([0,0,0], [1,1,1])
im_lb = dict(im=img, lb=None)
im_lb = trans(im_lb)
input = im_lb['im']

# 模型推理
input = input.unsqueeze(dim=0)
pred = resnet(input)[0]
pred = pred.detach().cpu().numpy()
pred = np.uint8(pred * 255)

# 输出结果
cv2.imwrite('predict.png', pred)
