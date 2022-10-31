# %%
import matplotlib.pyplot as plt
import seaborn as sns
import torch as t
from torch import nn
from torch.nn import functional
from einops import reduce, repeat, rearrange
import plotly.express as px
import numpy as np
from fancy_einsum import einsum
from math import sqrt

# %%

def single_head_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
    '''
    Should return the results of self-attention (see the "Self-Attention in Detail" section of the Illustrated Transformer).

    With this function, you can ignore masking.

    Q: shape (batch_size, seq_len, head_size)
    K: shape (batch_size, seq_len, head_size)
    V: shape (batch_size, seq_len, head_size_v)

    It'll most likely be the case that head_size_v = head_size, but it's not guaranteed

    Return: shape (batch_size, seq_len, head_size_v)
    '''
    
    assert Q.shape == K.shape
    assert V.shape[:2] == Q.shape[:2]
    
    attention_scores = einsum('b seq_len_q head_size, b seq_len_k head_size -> b seq_len_q seq_len_k', Q, K)

    attention_probabilities = functional.softmax(attention_scores / sqrt(Q.shape[-1]), dim=-1) 

    return einsum('b seq_len_k head_size_v, b seq_len_q seq_len_k -> b seq_len_q head_size_v', V, attention_probabilities)


# %%

def single_head_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
    '''
    Should return the results of masked self-attention.

    See "The Decoder Side" section of the Illustrated Transformer for an explanation of masking.

    Q: shape (batch_size, seq_len, head_size)
    K: shape (batch_size, seq_len, head_size)
    V: shape (batch_size, seq_len, head_size_v)

    It'll most likely be the case that head_size_v = head_size, but it's not guaranteed

    Return: shape (batch_size, seq_len, head_size_v)
    '''
    
    assert Q.shape == K.shape
    assert V.shape[:2] == Q.shape[:2]
    seq_len = Q.shape[-2]
    
    attention_scores = einsum('b seq_len_q head_size, b seq_len_k head_size -> b seq_len_q seq_len_k', Q, K)

    mask = t.zeros(size=(seq_len, seq_len))
    for i in range(seq_len):
        mask[..., i, i+1:] = -t.inf
    
    attention_scores += mask
    attention_probabilities = functional.softmax(attention_scores / sqrt(Q.shape[-1]), dim=-1) 

    return einsum('b seq_len_k head_size_v, b seq_len_q seq_len_k -> b seq_len_q head_size_v', V, attention_probabilities)


# %%

def multihead_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor, num_heads: int):
    '''
    Implements multihead masked attention on the matrices Q, K and V.

    Q: shape (batch, seq, nheads*headsize)
    K: shape (batch, seq, nheads*headsize)
    V: shape (batch, seq, nheads*headsize)
    '''

    Q = rearrange(Q, 'batch seq (nheads h_size) -> batch seq nheads h_size', nheads = num_heads)
    K = rearrange(K, 'batch seq (nheads h_size) -> batch seq nheads h_size', nheads = num_heads)
    V = rearrange(V, 'batch seq (nheads h_size) -> batch seq nheads h_size', nheads = num_heads)

    seq_len = Q.shape[1]
    head_size = Q.shape[-1]

    attention_scores = einsum('b seq_q nheads h_size, b seq_k nheads h_size -> b nheads seq_q seq_k', Q, K)

    mask = t.zeros(size=(seq_len, seq_len))
    for i in range(seq_len):
        mask[..., i, i+1:] = -t.inf

    attention_scores += mask
    attention_probabilities = functional.softmax(attention_scores / sqrt(head_size), dim=-1) 

    values = einsum('b seq_k nheads h_size, b nheads seq_q seq_k -> b nheads seq_q h_size', V, attention_probabilities)
    return rearrange(values, 'b nheads seq_q h_size -> b seq_q nheads (nheads h_size)')

# If the above is wrong, the most likely culprits are the following:
# the ordering of nheads and h_size in the parentheses in the rearrange on the last line
# or of nheads and h_size in the parentheses in the rearranges in the first three lines

# %%

dtype = t.float
device = 'cuda' if t.cuda.is_available() else 'cpu'

class MultiheadMaskedAttention(nn.Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):

        super().__init__()

        self.num_heads = num_heads

        W_QKV = nn.Linear(in_features=hidden_size, out_features=3*hidden_size, dtype=dtype, device=device)
        W_O = nn.Linear(in_features=hidden_size, out_features=hidden_size, dtype=dtype, device=device)


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        '''

        hidden_size = x.shape[-1]

        QKV = self.W_QKV(x)

        # The below should also work instead of the following 3 lines
        # Q, K, V = rearrange(QKV, 'b seq (three hidden_size) -> three b seq hidden_size', three=3)

        Q = QKV[..., :hidden_size]
        K = QKV[..., hidden_size:2*hidden_size]
        V = QKV[..., 2*hidden_size:]

        attention_values = multihead_masked_attention(Q, K, V, self.num_heads)

        return self.W_O(attention_values)


# If the above is wrong, the most likely culprits are the following: 
# Maybe the second line of __init__ shouldn't have out_features = 3 * hidden_size