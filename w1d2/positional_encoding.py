# %%
import matplotlib.pyplot as plt
import seaborn as sns
import torch as t
from torch import nn
from torch import Tensor
from einops import reduce, repeat, rearrange
import plotly.express as px
import numpy as np

# %%

class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, embedding_dim: int):

        super().__init__()

        graph1 = t.arange(max_seq_len)
        graph2 = 1 / 1e4 ** (t.arange(0,embedding_dim,step=2) / embedding_dim)

        graph = t.outer(graph1, graph2)
        graph = rearrange(t.cat([t.sin(graph), t.cos(graph)], dim=1), 'L (d1 d2) -> L (d2 d1)', d1=2)

        self.register_buffer('PE_matrix', graph)


    def forward(self, x: Tensor) -> Tensor:
        '''
        x: shape (batch, seq_len, embedding_dim)
        '''

        return x + self.PE_matrix

# %%
