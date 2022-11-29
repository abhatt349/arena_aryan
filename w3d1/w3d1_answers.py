#%%

import torch as t
from torch import nn, optim
import numpy as np
from typing import Callable, Iterable, Optional

import utils

MAIN = (__name__ == '__main__')
device = "cuda" if t.cuda.is_available() else "cpu"

# %%

def rosenbrocks_banana(x: t.Tensor, y: t.Tensor, a=1, b=100) -> t.Tensor:
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1

if MAIN:
    x_range = [-2, 2]
    y_range = [-1, 3]
    fig = utils.plot_fn(rosenbrocks_banana, x_range, y_range, log_scale=True)
    fig.show()

# %%

def opt_fn_with_sgd(fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100):
    '''
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    '''
    assert xy.requires_grad
    
    out = t.zeros(size=(n_iters, 2), dtype=t.float, device=device)

    optimizer = optim.SGD(params=[xy], lr=lr, momentum=momentum)

    for iter in range(n_iters):

        out[iter,:] = xy.clone().detach()

        loss = fn(xy[0], xy[1])
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return out


# %%

if MAIN:

    xy = t.tensor([-1.5, 2.5], requires_grad=True)
    # print(opt_fn_with_sgd(rosenbrocks_banana, xy=xy))

    x_range = [-2, 2]
    y_range = [-1, 3]

    fig = utils.plot_optimization_sgd(opt_fn_with_sgd, rosenbrocks_banana, xy, x_range, y_range, lr=0.001, momentum=0.98, show_min=True)

    fig.show()

# %%

class SGD:

    params: list[t.nn.parameter.Parameter]

    def __init__(self, params: Iterable[t.nn.parameter.Parameter], lr: float, momentum: float = 0, weight_decay: float = 0):
        '''Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        '''
        
        self.params = list(params)
        self.num_params = len(self.params)
        self.lr = lr
        self.momentum = momentum
        self.wd = weight_decay
        self.g = [t.zeros_like(param) for param in self.params]

        self.t = 0

    def zero_grad(self) -> None:

        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:

        self.t += 1

        temp_g = [param.grad+self.wd*param.data for param in self.params]

        for i in range(self.num_params):
            self.g[i] = self.momentum*self.g[i] + temp_g[i]
            self.params[i] -= self.lr * self.g[i]


    def __repr__(self) -> str:
        # Should return something reasonable here, e.g. "SGD(lr=lr, ...)"
        return f'SGD(params={self.params}, lr={self.lr}, momentum={self.momentum}, weight_decay={self.wd}'

#%%

if MAIN:
    utils.test_sgd(SGD)

# %%

class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        alpha: float = 0,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
    ):
        '''Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
        '''
        
        self.params = list(params)
        self.num_params = len(self.params)
        self.lr = lr
        self.momentum = momentum
        self.wd = weight_decay
        self.alpha = alpha
        self.eps = eps
        self.v = [t.zeros_like(param) for param in self.params]
        self.b = [t.zeros_like(param) for param in self.params]

        self.t = 0

    def zero_grad(self) -> None:

        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:

        self.t += 1

        temp_g = [param.grad+self.wd*param.data for param in self.params]

        for i in range(self.num_params):

            self.v[i] = self.alpha * self.v[i] + (1-self.alpha) * temp_g[i]**2
            self.b[i] = self.momentum*self.b[i] + temp_g[i]/(t.sqrt(self.v[i])+self.eps)
        
            self.params[i] -= self.lr * self.b[i]


    def __repr__(self) -> str:
        # Should return something reasonable here, e.g. "SGD(lr=lr, ...)"
        return f'RMSProp(params={self.params}, lr={self.lr}, alpha={self.alpha}, momentum={self.momentum}, weight_decay={self.wd}'

#%%

if MAIN:
    utils.test_rmsprop(RMSprop)

# %%

class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        '''
        
        self.params = list(params)
        self.num_params = len(self.params)
        self.lr = lr
        self.momentum = betas[0]
        self.wd = weight_decay
        self.alpha = betas[1]
        self.eps = eps
        self.v = [t.zeros_like(param) for param in self.params]
        self.m = [t.zeros_like(param) for param in self.params]

        self.t = 0

    
    def zero_grad(self) -> None:

        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:

        self.t += 1

        temp_g = [param.grad+self.wd*param.data for param in self.params]

        for i in range(self.num_params):

            self.v[i] = self.alpha * self.v[i] + (1-self.alpha) * temp_g[i]**2
            self.m[i] = self.momentum * self.m[i] + (1-self.momentum) * temp_g[i]

            unbiased_m = self.m[i] / (1-self.momentum**self.t)
            unbiased_v = self.v[i] / (1-self.alpha**self.t)
            update = unbiased_m / (t.sqrt(unbiased_v)+self.eps)
        
            self.params[i] -= self.lr * update 


    def __repr__(self) -> str:
        # Should return something reasonable here, e.g. "SGD(lr=lr, ...)"
        return f'Adam(params={self.params}, lr={self.lr}, beta_1={self.momentum}, beta_2={self.alpha}, weight_decay={self.wd}'


#%%

if MAIN:
    utils.test_adam(Adam)
    
# %%

def opt_fn(fn: Callable, xy: t.Tensor, optimizer_class, optimizer_kwargs, n_iters: int = 100):
    '''Optimize the a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
    optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    '''
    assert xy.requires_grad
    
    out = t.zeros(size=(n_iters, 2), dtype=t.float, device=device)

    optimizer = optimizer_class([xy], **optimizer_kwargs)

    for iter in range(n_iters):

        out[iter,:] = xy.clone().detach()

        loss = fn(xy[0], xy[1])
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return out

# %%

if MAIN:
    xy = t.tensor([-1.5, 2.5], requires_grad=True)
    x_range = [-2, 2]
    y_range = [-1, 3]
    optimizers = [
        (SGD, dict(lr=1e-3, momentum=0.98)),
        (SGD, dict(lr=5e-4, momentum=0.98)),
    ]

    fig = utils.plot_optimization(opt_fn, rosenbrocks_banana, xy, optimizers, x_range, y_range)

    fig.show()

# %%
