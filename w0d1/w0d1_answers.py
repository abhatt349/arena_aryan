# %%
import numpy as np
import torch as t
from torch import optim
from torch import nn
import plotly.express as px
import plotly.graph_objs as go
from typing import Optional, Callable
import ipywidgets as wg
from fancy_einsum import einsum
import math

import utils

# %%

def DFT_1d(arr: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Returns the DFT of the array `arr`, with an optional `inverse` argument.
    """
    N = arr.shape[0]
    dft_mat = np.outer(np.arange(N), np.arange(N)) 
    dft_mat = (2j * np.pi / N) * (1 if inverse else -1) * dft_mat
    dft_mat = np.exp(dft_mat) * (1/N if inverse else 1)
    return dft_mat @ arr 

utils.test_DFT_func(DFT_1d)

# %%
def my_DFT_test(DFT_func):
    x = np.array([1, 2-1j, -1j, -1+2j])
    y_actual = DFT_func(x)
    y_expected = np.array([2, -2-2j, -2j, 4+4j])

    np.testing.assert_allclose(y_actual, y_expected) 

my_DFT_test(DFT_1d)
# %%

def integrate_function(func: Callable, x0: float, x1: float, n_samples: int = 1000):
    """
    Calculates the approximation of the Riemann integral of the function `func`, 
    between the limits x0 and x1.

    You should use the Left Rectangular Approximation Method (LRAM).
    """

    x_samples = np.linspace(start=x0, stop=x1, num=n_samples, endpoint=False)
    # y_samples = np.array([func(x) for x in x_samples])
    y_samples = func(x_samples)
    return np.sum(y_samples)*(x1-x0)/n_samples

utils.test_integrate_function(integrate_function)

# %%
def integrate_product(func1: Callable, func2: Callable, x0: float, x1: float):
    """
    Computes the integral of the function x -> func1(x) * func2(x).
    """

    return integrate_function(lambda x: func1(x)*func2(x), x0, x1)

utils.test_integrate_product(integrate_product)

# %%
def calculate_fourier_series(func: Callable, max_freq: int = 50):
    """
    Calculates the fourier coefficients of a function, 
    assumed periodic between [-pi, pi].

    Your function should return ((a_0, A_n, B_n), func_approx), where:
        a_0 is a float
        A_n, B_n are lists of floats, with n going up to `max_freq`
        func_approx is the fourier approximation, as described above
    """

    a_0 = integrate_function(func, -np.pi, np.pi) / np.pi
    A_n = np.zeros(max_freq)
    B_n = np.zeros(max_freq)
    for i in range(max_freq):
        A_n[i] = integrate_product(func, lambda x: np.cos((i+1)*x), -np.pi, np.pi) / np.pi
        B_n[i] = integrate_product(func, lambda x: np.sin((i+1)*x), -np.pi, np.pi) / np.pi

    def func_approx(x):
        if max_freq == 0:
            return a_0/2
        input = np.arange(1,max_freq+1) * x
        return a_0/2 + A_n @ np.cos(input) + B_n @ np.sin(input)

    func_approx = np.vectorize(func_approx)
    
    return ((a_0, A_n, B_n), func_approx)


step_func = lambda x: np.sin(5*x)-.5*np.cos(-2*x)
utils.create_interactive_fourier_graph(calculate_fourier_series, func = step_func)
# %%

NUM_FREQUENCIES = 10
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

x = np.linspace(-np.pi, np.pi, 2000)
y = TARGET_FUNC(x)

x_cos = np.array([np.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = np.array([np.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = np.random.randn()
A_n = np.random.randn(NUM_FREQUENCIES)
B_n = np.random.randn(NUM_FREQUENCIES)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):

    # TODO: compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    # raise Exception("Not yet implemented.")
    y_pred = a_0/2 + einsum('n x, n -> x', x_cos, A_n) + einsum('n x, n -> x', x_sin, B_n)

    # TODO: compute `loss`, which is the sum of squared error between `y` and `y_pred`
    # raise Exception("Not yet implemented.")
    loss = np.square(y-y_pred).sum()

    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])
        y_pred_list.append(y_pred)

    # TODO: compute gradients of coeffs with respect to `loss`
    # raise Exception("Not yet implemented.")
    a_0_grad = (y_pred-y).sum()
    A_n_grad = 2 * einsum('x, n x -> n', y_pred-y, x_cos)
    B_n_grad = 2 * einsum('x, n x -> n', y_pred-y, x_sin)


    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)
    # raise Exception("Not yet implemented.")
    a_0 -= a_0_grad * LEARNING_RATE
    A_n -= A_n_grad * LEARNING_RATE
    B_n -= B_n_grad * LEARNING_RATE

utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)

# %%



NUM_FREQUENCIES = 10
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

x = t.linspace(-t.pi, t.pi, 2000)
y = TARGET_FUNC(x)


x_cos = t.stack([t.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = t.stack([t.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])
# x_cos = t.cos(t.outer(t.arange(1,NUM_FREQUENCIES+1), x)).shape
# x_sin = t.sin(t.outer(t.arange(1,NUM_FREQUENCIES+1), x)).shape

a_0 = t.randn(1)[0]
A_n = t.randn(NUM_FREQUENCIES)
B_n = t.randn(NUM_FREQUENCIES)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):

    # TODO: compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    # raise Exception("Not yet implemented.")
    y_pred = a_0/2 + einsum('n x, n -> x', x_cos, A_n) + einsum('n x, n -> x', x_sin, B_n)

    # TODO: compute `loss`, which is the sum of squared error between `y` and `y_pred`
    # raise Exception("Not yet implemented.")
    loss = t.square(y-y_pred).sum()

    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0, A_n.numpy(), B_n.numpy()])
        y_pred_list.append(y_pred.numpy().copy())

    # TODO: compute gradients of coeffs with respect to `loss`
    # raise Exception("Not yet implemented.")
    a_0_grad = (y_pred-y).sum()
    A_n_grad = 2 * einsum('x, n x -> n', y_pred-y, x_cos)
    B_n_grad = 2 * einsum('x, n x -> n', y_pred-y, x_sin)


    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)
    # raise Exception("Not yet implemented.")
    a_0 -= a_0_grad * LEARNING_RATE
    A_n -= A_n_grad * LEARNING_RATE
    B_n -= B_n_grad * LEARNING_RATE

utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)

# %%



NUM_FREQUENCIES = 10
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

x = t.linspace(-t.pi, t.pi, 2000, dtype=t.float)
y = TARGET_FUNC(x)


x_cos = t.stack([t.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = t.stack([t.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])
# x_cos = t.cos(t.outer(t.arange(1,NUM_FREQUENCIES+1), x)).shape
# x_sin = t.sin(t.outer(t.arange(1,NUM_FREQUENCIES+1), x)).shape

a_0 = t.randn(1, dtype=t.float, requires_grad=True)
A_n = t.randn(NUM_FREQUENCIES, dtype=t.float, requires_grad=True)
B_n = t.randn(NUM_FREQUENCIES, dtype=t.float, requires_grad=True)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):

    # TODO: compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = a_0/2 + einsum('n x, n -> x', x_cos, A_n) + einsum('n x, n -> x', x_sin, B_n)

    # TODO: compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = t.square(y-y_pred).sum()

    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0.detach().numpy().item(), A_n.detach().numpy(), B_n.detach().numpy()])
        y_pred_list.append(y_pred.detach().numpy().copy())

    # TODO: compute gradients of coeffs with respect to `loss`
    loss.backward()

    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)
    with t.no_grad():
        for coeff in [a_0, A_n, B_n]:
            if coeff.grad is not None:
                coeff -= LEARNING_RATE * coeff.grad
            coeff.grad = None


utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)

# %%

NUM_FREQUENCIES = 10
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

x = t.linspace(-math.pi, math.pi, 2000, dtype=t.float)
y = TARGET_FUNC(x)


x_cos = t.stack([t.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)]).T
x_sin = t.stack([t.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)]).T
x_all = t.cat([x_cos, x_sin], dim=1)

y_pred_list = []
coeffs_list = []

model = nn.Sequential(nn.Linear(2 * NUM_FREQUENCIES, 1), nn.Flatten(0, 1))

for step in range(TOTAL_STEPS):

    # TODO: compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = model(x_all)

    # TODO: compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = t.square(y-y_pred).sum()

    if step % 100 == 0:
        print(f"{loss = :.2f}")
        weights = list(model.parameters())[0].squeeze(dim=0).detach().numpy().copy()
        bias = list(model.parameters())[1].detach().numpy().copy().item()
        coeffs_list.append((bias, weights[:NUM_FREQUENCIES], weights[NUM_FREQUENCIES:]))
        y_pred_list.append(y_pred.detach().numpy().copy())

    # TODO: compute gradients of coeffs with respect to `loss`
    loss.backward()

    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)
    with t.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param -= param.grad * LEARNING_RATE
    model.zero_grad()


utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)


# %%



NUM_FREQUENCIES = 10
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

x = t.linspace(-math.pi, math.pi, 2000, dtype=t.float)
y = TARGET_FUNC(x)


x_cos = t.stack([t.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)]).T
x_sin = t.stack([t.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)]).T
x_all = t.cat([x_cos, x_sin], dim=1)

y_pred_list = []
coeffs_list = []

model = nn.Sequential(nn.Linear(2 * NUM_FREQUENCIES, 1), nn.Flatten(0, 1))
optimizer = optim.SGD(model.parameters(), LEARNING_RATE)

for step in range(TOTAL_STEPS):

    # TODO: compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = model(x_all)

    # TODO: compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = t.square(y-y_pred).sum()

    if step % 100 == 0:
        print(f"{loss = :.2f}")
        weights = list(model.parameters())[0].squeeze(dim=0).detach().numpy().copy()
        bias = list(model.parameters())[1].detach().numpy().copy().item()
        coeffs_list.append((bias, weights[:NUM_FREQUENCIES], weights[NUM_FREQUENCIES:]))
        y_pred_list.append(y_pred.detach().numpy().copy())

    # TODO: compute gradients of coeffs with respect to `loss`
    loss.backward()

    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)
    with t.no_grad():
        optimizer.step()
    model.zero_grad()


utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)


# %%


