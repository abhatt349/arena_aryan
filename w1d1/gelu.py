# %%
import matplotlib.pyplot as plt
import seaborn as sns
import torch as t
from einops import reduce, repeat, rearrange
import plotly.express as px
import numpy as np
from torch.distributions.normal import Normal
from math import sqrt

# %%

cdf = Normal(0,1).cdf

start = -10
stop = 10
n_steps = 5001
x = t.linspace(start, stop, n_steps)

# %%

gelu = x*cdf(x)
approx1 = x/2*(1+t.tanh(sqrt(2/t.pi)*(x+0.044715*(x**3))))
approx2 = x*t.sigmoid(1.702*x)

plt.plot(gelu, label='GeLU')
plt.plot(approx1, label='approx1')
# plt.plot(approx2, label='approx2')

# plt.plot(approx1-gelu, label='approx1 error')
# plt.plot(approx2-gelu, label='approx2 error')

plt.legend()
plt.show()

# %%
