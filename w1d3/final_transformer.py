# %%
from ast import Mult
import matplotlib.pyplot as plt
from regex import D
import seaborn as sns
import torch as t
from torch import nn, norm
from torch import Tensor
from torch.nn import functional
from einops import reduce, repeat, rearrange
import plotly.express as px
import numpy as np
from fancy_einsum import einsum
from math import sqrt
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from typing import Callable
from tqdm.notebook import tqdm_notebook
from w0d3_utils import plot_loss_and_accuracy
from collections import OrderedDict

# %%

dtype = t.float
device = 'cuda' if t.cuda.is_available() else 'cpu'

# %%

@dataclass(frozen=True)
class TransformerConfig:
    '''Constants used throughout your decoder-only transformer model.'''

    num_layers: int
    num_heads: int
    vocab_size: int
    hidden_size: int
    max_seq_len: int
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05


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
    return rearrange(values, 'b nheads seq_q h_size -> b seq_q (nheads h_size)')

# If the above is wrong, the most likely culprits are the following:
# the ordering of nheads and h_size in the parentheses in the rearrange on the last line
# or of nheads and h_size in the parentheses in the rearranges in the first three lines

# If the above is wrong, the most likely culprits are the following:
# the ordering of nheads and h_size in the parentheses in the rearrange on the last line
# or of nheads and h_size in the parentheses in the rearranges in the first three lines

# %%

class MultiheadMaskedAttention(nn.Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):

        super().__init__()

        self.num_heads = num_heads

        self.W_QKV = nn.Linear(in_features=hidden_size, out_features=3*hidden_size, dtype=dtype, device=device)
        self.W_O = nn.Linear(in_features=hidden_size, out_features=hidden_size, dtype=dtype, device=device)


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

        seq_len = x.shape[1]
        return x + self.PE_matrix[:seq_len, :] #  type: ignore


# %%

class MLP(nn.Module):

    def __init__(self, config: TransformerConfig):

        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(config.hidden_size, 4*config.hidden_size, dtype=dtype, device=device)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4*config.hidden_size, config.hidden_size, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:
        
        x = self.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))

        return x


class DecoderBlock(nn.Module):

    def __init__(self, config: TransformerConfig):

        super().__init__()
        self.config = config

        self.attention = MultiheadMaskedAttention(self.config.hidden_size, self.config.num_heads)  # type: ignore
        self.ln1 = nn.LayerNorm(normalized_shape=config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)
        self.ln2 = nn.LayerNorm(normalized_shape=config.hidden_size, eps=config.layer_norm_epsilon)


    def forward(self, x: t.Tensor) -> t.Tensor:

        x = self.ln1(self.attention(x)) + x
        x = self.ln2(self.mlp(x)) + x

        return x


class DecoderOnlyTransformer(nn.Module):

    def __init__(self, config: TransformerConfig):

        super().__init__()

        self.token_emb = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.pos_emb = PositionalEncoding(config.max_seq_len, config.hidden_size)
        self.dropout = nn.Dropout(p=config.dropout)
        # self.blocks = nn.Sequential({'block '+str(i):DecoderBlock(config) for i in range(config.num_layers)}) # type: ignore
        self.blocks = nn.Sequential(OrderedDict(
            [(f'block {i}',DecoderBlock(config)) for i in range(config.num_layers)]
            )) # type: ignore
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size)


    def forward(self, x: t.Tensor) -> t.Tensor:
        
        embedding = self.token_emb(x.to(dtype=t.long))
        embedding = self.pos_emb(embedding)
        embedding = self.dropout(embedding)
        embedding = self.blocks(embedding)
        embedding = self.layer_norm(embedding)

        logits = einsum('vocab d, batch seq d -> batch seq vocab', self.token_emb.weight, embedding)

        return logits

# %%

class CustomTextDataset(Dataset):
    def __init__(self, text, labels):
        self.labels = labels
        self.text = text

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.text[idx]
        sample = (text, label)
        return sample


# %%

train_set_size = int(1e4)
test_set_size = int(1e3)
seq_len = 6
batch_size = 128
num_digits = 10
embed_dim = 10
hidden_size = 5

config = TransformerConfig(num_layers=2, 
                           num_heads=embed_dim//hidden_size, 
                           vocab_size=num_digits, 
                           hidden_size=embed_dim, 
                           max_seq_len=seq_len
                          )

train_x = t.randint(low=0, high=num_digits, size=(train_set_size, seq_len), dtype=dtype, device=device)
dataset = CustomTextDataset(train_x, train_x.flip(dims=(1,)).to(dtype=t.long))
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

test_x = t.randint(low=0, high=num_digits, size=(test_set_size, seq_len), dtype=dtype, device=device)
dataset = CustomTextDataset(test_x, test_x.flip(dims=(1,)).to(dtype=t.long))
testloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# %%


epochs = 25
loss_fn = t.nn.CrossEntropyLoss()   

MODEL_FILENAME = "./w1d3_reversing_transformer.pt"

def train_transformer(trainloader: DataLoader, testloader: DataLoader, epochs: int, loss_fn: Callable, config: TransformerConfig) -> list:
    '''
    Returns tuple of (loss_list, accuracy_list), where accuracy_list contains the fraction of accurate classifications on the test set, at the end of each epoch.
    '''

    model = DecoderOnlyTransformer(config).to(device).train()
    optimizer = t.optim.Adam(model.parameters())
    loss_list = []
    accuracy_list = []

    for epoch in tqdm_notebook(range(epochs)):

        loss = None
        
        for (x, y) in tqdm_notebook(trainloader, leave=False):

            x = x.to(device)
            y = y.to(device)
            y = rearrange(y, 'batch seq_len -> (batch seq_len)')

            y_hat = model(x)
            y_hat = rearrange(y_hat, 'batch seq_len logit -> (batch seq_len) logit')

            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.item())
        
        for (x, y) in tqdm_notebook(testloader, leave=False):

            x = x.to(device)
            y = y.to(device)
            y = rearrange(y, 'batch seq_len -> (batch seq_len)')


            y_hat = model(x)
            y_hat = rearrange(y_hat, 'batch seq_len logit -> (batch seq_len) logit')
            preds = t.argmax(y_hat, dim=1)
            
            accuracy = (y == preds).to(float).mean().item()
            accuracy_list.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}") 

    print(f"Saving model to: {MODEL_FILENAME}")
    t.save(model, MODEL_FILENAME)
    return [loss_list, accuracy_list]

# %%

loss_list, accuracy_list = train_transformer(trainloader, testloader, epochs, loss_fn, config)

plot_loss_and_accuracy(loss_list, accuracy_list)

# %%
