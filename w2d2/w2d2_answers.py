# %%
import time

import torch as t
import transformers
import utils
import wandb
from einops import rearrange
from fancy_einsum import einsum
from functions_from_previous_days import *
from plotly.subplots import make_subplots
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.notebook import tqdm_notebook
from collections import OrderedDict
import transformers

MAIN = (__name__ == '__main__')
dtype = t.float
device = "cuda" if t.cuda.is_available() else "cpu"

# %%

# transformers.AutoModelForCausalLM.from_pretrained("gpt2")

# %%

def GPTmultihead_masked_attention(Q, K, V, num_heads, dropout):

    device = Q.device

    # Rearrange Q, K and V to separate the `headsize` dimension (because this is the one we take the inner product over)
    q = rearrange(Q, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)
    k = rearrange(K, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)
    v = rearrange(V, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)

    # Calculate attention scores as inner product of q and k, and divide by sqrt(headsize)
    batch, seq_len, nheads, headsize = q.shape
    attention_scores = einsum("batch seqQ nheads headsize, batch seqK nheads headsize -> batch nheads seqQ seqK", q, k) / (headsize ** 0.5)

    # Create the attention mask
    # Note we don't need to add batch and nheads, for broadcasting reasons
    # Also note you could do this with much less code using e.g. t.triu(t.ones(...)), but this way is more explicit
    q_idx = repeat(t.arange(seq_len), "seqQ -> seqQ seqK", seqK=seq_len)
    k_idx = repeat(t.arange(seq_len), "seqK -> seqQ seqK", seqQ=seq_len)
    # Any index positions with q<k should be masked (this prevents tokens "reading info from the future")
    mask = (q_idx >= k_idx).to(device)
    neg_inf = t.tensor(-1e6, dtype=attention_scores.dtype, device=device)
    attention_scores = t.where(mask, attention_scores, neg_inf)

    # Take softmax over the key dimension (i.e. each query index has a corresponding probability distribution over tokens in the sequence)
    attention_probabilities = attention_scores.softmax(dim=-1)
    attention_probabilities = dropout(attention_probabilities)

    # Get attention values by taking convex combination of value vectors according to the attention probabilities
    attention_values = einsum("batch nheads seqQ seqK, batch seqK nheads headsize -> batch seqQ nheads headsize", attention_probabilities, v)

    # Rearrange to combine the nheads and headsize dimensions
    return rearrange(attention_values, "batch seqQ nheads headsize -> batch seqQ (nheads headsize)")



class GPTMultiheadMaskedAttention(Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    dropout1: nn.Dropout
    dropout2: nn.Dropout

    def __init__(self, config: TransformerConfig):

        super().__init__()

        self.num_heads = config.num_heads
        hidden_size = config.hidden_size

        self.W_QKV = nn.Linear(in_features=hidden_size, out_features=3*hidden_size, dtype=dtype, device=device)  # type: ignore
        self.W_O = nn.Linear(in_features=hidden_size, out_features=hidden_size, dtype=dtype, device=device)  # type: ignore

        self.dropout1 = nn.Dropout(p=config.dropout)  # type: ignore
        self.dropout2 = nn.Dropout(p=config.dropout)  # type: ignore

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

        attention_values = GPTmultihead_masked_attention(Q, K, V, self.num_heads, self.dropout1)

        return self.dropout2(self.W_O(attention_values))

# If the above is wrong, the most likely culprits are the following: 
# Maybe the second line of __init__ shouldn't have out_features = 3 * hidden_size


# %%

class MLP(Module):

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


class GPTDecoderBlock(Module):

    def __init__(self, config: TransformerConfig):

        super().__init__()
        self.config = config

        self.ln1 = nn.LayerNorm(normalized_shape=config.hidden_size, eps=config.layer_norm_epsilon)  # type: ignore
        self.attention = GPTMultiheadMaskedAttention(self.config)
        self.ln2 = nn.LayerNorm(normalized_shape=config.hidden_size, eps=config.layer_norm_epsilon)  # type: ignore
        self.mlp = MLP(config)


    def forward(self, x: t.Tensor) -> t.Tensor:

        x = self.attention(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x

        return x


class GPT(Module):

    def __init__(self, config: TransformerConfig):

        super().__init__()

        self.token_emb = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.pos_emb = nn.Embedding(num_embeddings=config.max_seq_len, embedding_dim=config.hidden_size)
        self.dropout = nn.Dropout(p=config.dropout)
        # self.blocks = nn.Sequential({'block '+str(i):DecoderBlock(config) for i in range(config.num_layers)}) # type: ignore
        self.blocks = nn.Sequential(OrderedDict(
            [(f'block {i}',GPTDecoderBlock(config)) for i in range(config.num_layers)]
            )) # type: ignore
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size)


    def forward(self, x: t.Tensor) -> t.Tensor:
        
        pos_vec = t.arange(x.shape[1], dtype=t.long, device=x.device)

        embedding = self.token_emb(x.to(dtype=t.long)) + self.pos_emb(pos_vec)
        embedding = self.dropout(embedding)
        embedding = self.blocks(embedding)
        embedding = self.layer_norm(embedding)

        logits = einsum('vocab d, batch seq d -> batch seq vocab', self.token_emb.weight, embedding)

        return logits

# %%

if MAIN:
    config = TransformerConfig(
        num_layers = 12,
        num_heads = 12,
        vocab_size = 50257,
        hidden_size = 768,
        max_seq_len = 1024,
        dropout = 0.1,
        layer_norm_epsilon = 1e-05,
        print_param_count = False
    )


    my_gpt = GPT(config).train()
    gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2").train()

    utils.print_param_count(my_gpt, gpt)

# %%

def copy_weights_from_gpt(my_gpt: GPT, gpt) -> GPT:
    '''
    Copy over the weights from gpt to your implementation of gpt.

    gpt should be imported using: 
        gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

    Returns your gpt model, with weights loaded in.

    You might find the function `copy_weights` from w0d3 helpful as a template.
    '''

    mydict = dict(my_gpt.named_parameters())
    pretraineddict = dict(gpt.named_parameters())

    # Check the number of params/buffers is correct
    assert len(list(mydict)) == len(list(pretraineddict)), "Number of layers is wrong. Have you done the prev step correctly?"

    # Initialise an empty dictionary to store the correct key-value pairs
    state_dict_to_load = {}

    for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items()):
        
        if (len(myvalue.shape) > 1) and ('block' in mykey):
            pretrainedvalue = pretrainedvalue.T

        state_dict_to_load[mykey] = pretrainedvalue


    my_gpt.load_state_dict(state_dict_to_load)   # type: ignore
    return my_gpt


# %%

if MAIN:
    
    my_gpt = copy_weights_from_gpt(my_gpt, gpt)

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    utils.test_load_pretrained_weights(gpt, tokenizer)
    utils.test_load_pretrained_weights(my_gpt, tokenizer)


    # %%
