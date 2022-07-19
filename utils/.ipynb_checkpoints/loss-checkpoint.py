import torch.nn as nn
import torch

# MSE Loss
class MSE_Loss(nn.Module):
    def __init__(self) -> None:
        super(MSE_Loss, self ).__init__() 
        self.loss =  nn.MSELoss(reduction = 'mean')

    def forward(self, logits, labels):
        return self.loss(logits, labels)


# CrossEntropyLoss
class CE_Loss(nn.Module):
    def __init__(self, ignore_lb =255) -> None:
        super(CE_Loss, self ).__init__() 
        self.loss = nn.CrossEntropyLoss(reduction = 'mean', ignore_index=ignore_lb)
       
    
    def forward(self, logits, labels):
        return self.loss(logits, labels)

# online hard example miniing CrossEntropyLoss
class OhemCELoss(nn.Module):
    def __init__(self, thresh = 0.7, ignore_lb =255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)