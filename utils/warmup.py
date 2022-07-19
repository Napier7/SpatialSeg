from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# todo: fixed 'constant' 'exp'
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_ratio: target learning rate = fn(base lr, warmup_ratio) 
        warmup_iters: target learning rate is reached at warmup_iters, gradually
        after_scheduler: after target_iter, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, warmup_ratio, warmup_iters, after_scheduler=None):
        self.warmup_ratio = warmup_ratio
        self.warmup_iters = warmup_iters
        self.after_scheduler = after_scheduler
        self.last_iter = 1
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_iter > self.warmup_iters:
            if self.after_scheduler:
                return self.after_scheduler.get_lr()
            return [base_lr * self.warmup_ratio for base_lr in self.base_lrs]

        else:
            k = (1 - self.last_iter / self.warmup_iters) * (1 - self.warmup_ratio)
            return [base_lr * (1 - k)  for base_lr in self.base_lrs]

    def step_warmup(self, iter=None):
        if iter is None:
            iter = self.last_iter + 1
        self.last_iter = iter if iter != 0 else 1  
        
        if self.last_iter <= self.warmup_iters:
            k = (1 - self.last_iter / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [base_lr * (1 - k)  for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if iter is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(iter - self.warmup_iters)

   
    def step(self, iter=None):
            if self.last_iter + 1 > self.warmup_iters and self.after_scheduler:
                if iter is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(iter - self.warmup_iters)
    
            else:
                self.step_warmup(iter)