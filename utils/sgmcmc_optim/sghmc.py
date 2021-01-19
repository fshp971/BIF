import numpy as np
import torch


''' References:
    [1] https://arxiv.org/pdf/1402.4102.pdf
    [2] https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
'''
class SGHMC(torch.optim.Optimizer):
    def __init__(self, params, lr, alpha, lr_decay=1.0):
        defaults = dict(lr=lr, alpha=alpha, lr_decay=lr_decay)
        super(SGHMC, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr_decay = group['lr_decay']
            lr = group['lr'] * lr_decay
            alpha = group['alpha'] * np.sqrt(lr_decay)

            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if 'velocity' not in param_state:
                    param_state['velocity'] = p.data.clone().normal_(0, np.sqrt(lr))
                    p.add_(param_state['velocity'])
                    continue

                velocity = param_state['velocity']

                v_grad = velocity.clone().normal_(0, np.sqrt(2*alpha*lr))
                v_grad.add_(p.grad, alpha= -lr)
                v_grad.add_(velocity, alpha= -alpha)
                velocity.add_(v_grad)

                p.add_(velocity)

        return loss

    def set_learning_rate(self, lr):
        for group in self.param_groups:
            group['lr'] = lr
