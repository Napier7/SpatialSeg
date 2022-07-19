import os
import torch
import torch.optim as optim
from utils.lr_scheduler import PolynomialLRDecay
from utils.warmup import GradualWarmupScheduler
from model.net import Net
from model import get_model
from utils.loss import CE_Loss, OhemCELoss
from utils.dataloader import getDataLoader
from tqdm import tqdm
from configs import set_cfg_from_file
import wandb

# get cfg
cfg, cfg_dic = set_cfg_from_file('configs/cfg.py')
# set cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# wandb
wandb.init(project="test-project", entity="m78-planet", config = cfg_dic)

# split params
def split_params(module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay

# save checkpoint
def save_checkpoint(model, opt, iter, train_loss, val_loss):
    root = cfg.train.checkpoint_root
    if not os.path.exists(root):
        os.makedirs(root)
    checkpoint = {
        "net": model.state_dict(),
        'optimizer':opt.state_dict(),
        "iter": iter
    }
    torch.save(checkpoint, os.path.join(root, 'Iter%d-Train_Loss%.4f-Val_Loss%.4f.pth'%(
        (iter), 
        (train_loss),
        (val_loss)
    )))

# get learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# set loss function
def set_loss():
#     Loss = CE_Loss()
    Loss = OhemCELoss()
    return Loss

def trainloop(model, loader, iter, optim = None, mode = 'train'):
    model.train()
    Loss = set_loss()
    total_loss = 0
    interval_iter = 0
    inter_iters = min(cfg.train.max_iters - iter, cfg.train.interval)

    pbar = tqdm(
        total = inter_iters,
        # desc =  f'Iter {iter}/{cfg.train.max_iters} {mode}',
    )
    while interval_iter < inter_iters:
        for index, batch in enumerate(loader):
            img, label = batch

            if cfg.train.cuda:
                img = img.to(device)
                label = label.to(device)
            
            # caculate loss
            logit = model(img)
            loss = Loss(logit, label)
            total_loss += loss.item()

            # lr step
            lr_scheduler.step()

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # update iter
            interval_iter += 1
            iter += 1

            # print
            pbar.set_postfix({
                'loss': total_loss / (interval_iter),
                'lr'  : get_lr(optim)
            })
            pbar.set_description(f'Iter {iter}/{cfg.train.max_iters} {mode}')

            # update pbar
            pbar.update(1)

            # end 
            if interval_iter >= inter_iters:
                break

    return total_loss / inter_iters

def valloop(model, loader, iter, optim = None, mode = 'train'):
    model.eval()
    Loss = set_loss()
    total_loss = 0
    pbar = tqdm(
        total = len(loader),
        # desc =  f'Iter {iter}/{cfg.train.max_iters} {mode}',
    )
    with torch.no_grad():
        for index, batch in enumerate(loader):
            img, label = batch

            if cfg.train.cuda:
                img = img.to(device)
                label = label.to(device)
            
            # caculate loss
            logit = model(img)
            loss = Loss(logit, label)
            total_loss += loss.item()

            
            # print
            pbar.set_postfix({
                'loss': total_loss / (index + 1),
            })
            pbar.set_description(f'Iter {iter}/{cfg.train.max_iters} {mode}')

            # update pbar
            pbar.update(1)
    return total_loss / len(loader)

if __name__ == '__main__':
    ## dataloader
    train_dataloader = getDataLoader(cfg, mode = 'train')
    val_dataloader = getDataLoader(cfg, mode = 'val')


    ## model
    model = get_model(cfg.model.type)
    model = model(n_class = cfg.model.n_class)
    if cfg.train.cuda:
        model.to(device)


    ## optimizer
    cfg_opt = cfg.optimizer
    param_decay, param_no_decay = split_params(model)
    optimizer = optim.AdamW([
        {'params': param_decay},
        {'params': param_no_decay, 'weight_decay': 0}
    ],  lr = cfg_opt.lr_start, weight_decay = cfg_opt.weight_decay, betas = cfg_opt.betas)
    after_scheduler = PolynomialLRDecay(optimizer, max_decay_steps = cfg.train.max_iters - 1500, end_learning_rate=0.0, power=1.0)
    lr_scheduler = GradualWarmupScheduler(optimizer, warmup_ratio = 1e-6, warmup_iters = 1500, after_scheduler = after_scheduler)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optimizer.zero_grad()
    optimizer.step()

    ## resume checkpoint
    start_iter = 0
    if cfg.train.resume:
        path = cfg.train.checkpoint
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint['iter']
        
    # train & val loops
    while start_iter < cfg.train.max_iters:
        # train
        model.mode = 'train'
        train_loss = trainloop(
            model = model, 
            loader = train_dataloader, 
            optim = optimizer,
            iter = start_iter, 
            mode = 'train'
        )
        start_iter += cfg.train.interval
        start_iter = start_iter if start_iter <= cfg.train.max_iters else cfg.train.max_iters
        
        # val
        model.mode = 'val'
        val_loss = valloop(
            model = model, 
            loader = val_dataloader, 
            iter = start_iter, 
            mode = 'val'
        )
 
        # checkpoint
        save_checkpoint(model, optimizer, start_iter, train_loss, val_loss)
        
        # visulization
        # todo: val iou
        wandb.log({"train loss": train_loss, 'val loss': val_loss})

    
    # shutdown
    os.system("shutdown")