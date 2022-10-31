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
    
    attention_scores = einsum('b seq_len_q head_size, b seq_len_k head_size -> b seq_len_q seq_len_k', Q, K)

    mask = t.zeros(size=(Q.shape[-1], K.shape[-1]))
    for i in range(Q.shape[-1]):
        mask[..., i, i+1:] = -t.inf
    
    attention_scores += mask
    attention_probabilities = functional.softmax(attention_scores / sqrt(Q.shape[-1]), dim=-1) 

    return einsum('b seq_len_k head_size_v, b seq_len_q seq_len_k -> b seq_len_q head_size_v', V, attention_probabilities)


# %%
