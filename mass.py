import torch
from torch.optim import Optimizer

class Mass(Optimizer):
    def __init__(self, params, lr=0.1, alpha=0.05, kappa_t=12):
        defaults = dict(lr=lr, alpha=alpha, kappa_t=kappa_t)
        super(Mass, self).__init__(params,defaults)

    def __setstate__(self, state):
        super(Mass, self).__setstate__(state)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            w_list = []
            v_list = []
            d_p_list = []
            lr = group['lr']
            alpha = group['alpha']
            kappa_t = group['kappa_t']
            delta = lr/alpha/kappa_t

            for p in group['params']:
                params_with_grad.append(p)
                w_list.append(p)
                v_list.append(p)
                d_p_list.append(p.grad)
                state = self.state[p]
            
            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                w = w_list[i]
                v = v_list[i]

                new_w = param - lr*d_p
                new_v = (1-alpha)*v + alpha*param - delta*d_p

                param.data = alpha/(1+alpha)*new_v.detach()  + 1/(1+alpha)*new_w.detach() 
                
                w_list[i] = new_w
                v_list[i] = new_v

            for p in params_with_grad:
                state = self.state[p]
            
        return loss