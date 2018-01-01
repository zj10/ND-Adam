import math
import torch
from torch.optim import Optimizer


class NDAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, vec_axes=None):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, vec_axes=vec_axes)
        super(NDAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ND-Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    if group['vec_axes']:
                        shape = list(p.size())
                        for i in group['vec_axes']:
                            shape[i] = 1
                        state['exp_avg_sq'] = torch.zeros(shape)
                        if torch.cuda.is_available():
                            state['exp_avg_sq'] = state['exp_avg_sq'].cuda()
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                if group['vec_axes']:
                    g_proj = grad * p.data
                    for i in group['vec_axes']:
                        g_proj = torch.sum(g_proj, i, True)
                    grad.add_(-g_proj * p.data)
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    g_sqr = grad * grad
                    for i in group['vec_axes']:
                        g_sqr = torch.sum(g_sqr, i, True)
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, g_sqr)
                else:
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                if group['vec_axes']:
                    norm = p.data * p.data
                    for i in group['vec_axes']:
                        norm = torch.sum(norm, i, True)
                    norm.sqrt_()
                    p.data.div_(norm)

        return loss
