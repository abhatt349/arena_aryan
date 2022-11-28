# %%
import math
import time
from collections import OrderedDict
from typing import List, Optional, Union

import plotly.express as px
import torch as t
import transformers
import utils
import wandb
from einops import rearrange
from fancy_einsum import einsum
from functions_from_previous_days import *
from plotly.subplots import make_subplots
from torch import nn, optim
from torch.nn import Module, Parameter  # type: ignore
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.notebook import tqdm_notebook

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

        self.fc1 = nn.Linear(config.hidden_size, 4*config.hidden_size, dtype=dtype, device=device)  # type: ignore
        self.gelu = nn.GELU()  # type: ignore
        self.fc2 = nn.Linear(4*config.hidden_size, config.hidden_size, dtype=dtype, device=device)  # type: ignore
        self.dropout = nn.Dropout(p=config.dropout)  # type: ignore

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

        self.token_emb = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)  # type: ignore
        self.pos_emb = nn.Embedding(num_embeddings=config.max_seq_len, embedding_dim=config.hidden_size)  # type: ignore
        self.dropout = nn.Dropout(p=config.dropout)  # type: ignore
        # self.blocks = nn.Sequential({'block '+str(i):DecoderBlock(config) for i in range(config.num_layers)}) # type: ignore
        self.blocks = nn.Sequential(OrderedDict(
            [(f'block {i}',GPTDecoderBlock(config)) for i in range(config.num_layers)]
            )) # type: ignore
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size)  # type: ignore


    def forward(self, x: t.Tensor) -> t.Tensor:
        
        if len(x.shape)==1:
            x = x.unsqueeze(dim=0)

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
    gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2").train()  # type: ignore

    # utils.print_param_count(my_gpt, gpt)

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
    
    my_gpt = copy_weights_from_gpt(my_gpt, gpt)  # type: ignore

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")  # type: ignore

    utils.test_load_pretrained_weights(gpt, tokenizer)  # type: ignore
    utils.test_load_pretrained_weights(my_gpt, tokenizer)


# %%

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):

        super().__init__()
        self.weight = Parameter(t.randn(size=(num_embeddings, embedding_dim), dtype=dtype, device=device)) 

    def forward(self, x: t.LongTensor) -> t.Tensor:
        '''For each integer in the input, return that row of the embedding.
        '''

        return self.weight[x]
            
            
    def extra_repr(self) -> str:
        return f"num_embeddings: {self.num_embeddings}, embedding_dim: {self.embedding_dim}"

if MAIN:
    utils.test_embedding(Embedding)
# %%

class GELU(nn.Module):

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x * (1+t.erf(x/math.sqrt(2))) / 2

if MAIN:
    utils.plot_gelu(GELU)

# %%

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-05, elementwise_affine: bool = True):
        
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(t.ones(normalized_shape, dtype=dtype, device=device))
            self.bias = Parameter(t.zeros(normalized_shape, dtype=dtype, device=device)) 


    def forward(self, x: t.Tensor) -> t.Tensor:
        
        dims = tuple(-t.arange(1,len(self.normalized_shape)+1))

        mean = x.mean(dim=dims, keepdims=True)   # type: ignore
        var = x.var(dim=dims, unbiased=False, keepdims=True)   # type: ignore

        x = (x - mean) / t.sqrt(var + self.eps)

        if self.elementwise_affine:
            x = x * self.weight + self.bias
        
        return x

utils.test_layernorm_mean_1d(LayerNorm)
utils.test_layernorm_mean_2d(LayerNorm)
utils.test_layernorm_std(LayerNorm)
utils.test_layernorm_exact(LayerNorm)
utils.test_layernorm_backward(LayerNorm)

# %%

class Dropout(nn.Module):

    def __init__(self, p: float):
        
        super().__init__()
        
        self.p = p

    def forward(self, x: t.Tensor) -> t.Tensor:
        
        if self.training:
            x = x*t.bernoulli(t.ones_like(x)*(1-self.p)) / (1-self.p)
        return x

utils.test_dropout_eval(Dropout)
utils.test_dropout_training(Dropout)

# %%

import transformers
import pandas as pd

gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

with pd.option_context("display.max_rows", None):
    display(pd.DataFrame([
        {"name": name, "shape": param.shape, "param count": param.numel()}
        for name, param in gpt.named_parameters()
    ]))


# %%

if MAIN:
    bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

# %%

if MAIN:
    print(tokenizer.tokenize('hello world'))

# %%

def BERTmultihead_masked_attention(Q, K, V, additive_attention_mask, num_heads):

    device = Q.device

    # Rearrange Q, K and V to separate the `headsize` dimension (because this is the one we take the inner product over)
    q = rearrange(Q, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)
    k = rearrange(K, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)
    v = rearrange(V, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)

    # Calculate attention scores as inner product of q and k, and divide by sqrt(headsize)
    batch, seq_len, nheads, headsize = q.shape
    attention_scores = einsum("batch seqQ nheads headsize, batch seqK nheads headsize -> batch nheads seqQ seqK", q, k) / (headsize ** 0.5)

    if additive_attention_mask is not None:
        attention_scores += additive_attention_mask

    # Take softmax over the key dimension (i.e. each query index has a corresponding probability distribution over tokens in the sequence)
    attention_probabilities = attention_scores.softmax(dim=-1)

    # Get attention values by taking convex combination of value vectors according to the attention probabilities
    attention_values = einsum("batch nheads seqQ seqK, batch seqK nheads headsize -> batch seqQ nheads headsize", attention_probabilities, v)

    # Rearrange to combine the nheads and headsize dimensions
    return rearrange(attention_values, "batch seqQ nheads headsize -> batch seqQ (nheads headsize)")


class MultiheadAttention(nn.Module):

    def __init__(self, config: TransformerConfig):
        
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_size = self.hidden_size // self.num_heads

        ## Store Q, K, and V separately to more easily copy weights
        # self.W_QKV = nn.Linear(self.hidden_size, 3*self.num_heads*self.head_size)
        
        self.W_Q = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_K = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_V = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_O = nn.Linear(self.num_heads*self.head_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)  # type: ignore

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor]) -> t.Tensor:
        
        ## Store Q, K, and V separately to more easily copy weights
        # QKV = self.W_QKV(x)
        # Q, K, V = t.split(QKV, self.num_heads*self.head_size, dim=-1)

        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)

        masked_attention_values = BERTmultihead_masked_attention(Q, K, V, additive_attention_mask, self.num_heads)
        output = self.W_O(masked_attention_values)

        return self.dropout(output)


class BERTBlock(nn.Module):

    def __init__(self, config):
        
        super().__init__()
        
        self.attention = MultiheadAttention(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        self.mlp = MLP(config)
        self.ln2 = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)


    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        additive_attention_mask: shape (batch, nheads=1, seqQ=1, seqK)
        '''

        x = self.ln1(x + self.attention(x, additive_attention_mask))
        x = self.ln2(x + self.mlp(x))

        return x


def make_additive_attention_mask(one_zero_attention_mask: t.Tensor, big_negative_number: float = -10000) -> t.Tensor:
    '''
    one_zero_attention_mask: 
        shape (batch, seq)
        Contains 1 if this is a valid token and 0 if it is a padding token.

    big_negative_number:
        Any negative number large enough in magnitude that exp(big_negative_number) is 0.0 for the floating point precision used.

    Out: 
        shape (batch, nheads=1, seqQ=1, seqK)
        Contains 0 if attention is allowed, big_negative_number if not.
    '''

    # one_zero_attention_mask = one_zero_attention_mask.to(dtype=t.float, device=device)
    additive_attention_mask = (1-one_zero_attention_mask)*big_negative_number

    return rearrange(additive_attention_mask, 'b s -> b 1 1 s')

utils.test_make_additive_attention_mask(make_additive_attention_mask)
print('Passed!')

# %%

class BertCommon(nn.Module):

    def __init__(self, config: TransformerConfig):
        
        super().__init__()

        self.tok_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_encoding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.typ_embedding = nn.Embedding(2, config.hidden_size)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.Sequential(*[BERTBlock(config) for i in range(config.num_layers)])  # type: ignore

    def forward(
        self,
        x: t.Tensor,
        one_zero_attention_mask: Optional[t.Tensor] = None,
        token_type_ids: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        '''
        x: (batch, seq) - the token ids
        one_zero_attention_mask: (batch, seq) - only used in training, passed to `make_additive_attention_mask` and used in the attention blocks.
        token_type_ids: (batch, seq) - only used for NSP, passed to token type embedding.
        '''
        
        additive_attention_mask = None if one_zero_attention_mask is None else make_additive_attention_mask(one_zero_attention_mask)

        tok_emb = self.tok_embedding(x)

        pos = t.arange(x.shape[1], dtype = x.dtype, device=x.device)
        pos_emb = self.pos_encoding(pos)
        
        if token_type_ids is None:
            token_type_ids = t.zeros(x.shape, dtype=t.int64, device = x.device)

        typ_emb = self.typ_embedding(token_type_ids)
        
        x = tok_emb + pos_emb + typ_emb

        x = self.ln1(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, additive_attention_mask)
        
        return x
        
# %%

class BertLanguageModel(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.bertcommon = BertCommon(config)
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.ln2 = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        self.tied_unembedding_bias = nn.Parameter(t.zeros(config.vocab_size, dtype=dtype, device=device))

    def forward(
        self,
        input_ids: t.Tensor,
        one_zero_attention_mask: Optional[t.Tensor] = None,
        token_type_ids: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        """Compute logits for each token in the vocabulary.

        Return: shape (batch, seq, vocab_size)
        """

        x = self.bertcommon(input_ids, one_zero_attention_mask, token_type_ids)
        x = self.ln2(self.gelu(self.fc(x)))

        x = einsum("batch seq hidden, vocab hidden -> batch seq vocab", x, self.bertcommon.tok_embedding.weight) + self.tied_unembedding_bias
        
        return x

#%%

if MAIN:

    config = TransformerConfig(
        num_layers = 12,
        num_heads = 12,
        vocab_size = 28996,
        hidden_size = 768,
        max_seq_len = 512,
        dropout = 0.1,
        layer_norm_epsilon = 1e-12
    )  # type: ignore

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

    my_bert = BertLanguageModel(config).train()
    bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")

    

# %%
def copy_weights_from_bert(my_bert: BertLanguageModel, bert: transformers.models.bert.modeling_bert.BertForMaskedLM) -> BertLanguageModel:
    '''
    Copy over the weights from bert to your implementation of bert.

    bert should be imported using: 
        bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")

    Returns your bert model, with weights loaded in.
    '''

    # FILL IN CODE: define a state dict from my_bert.named_parameters() and bert.named_parameters()

    my_bert_list = list(my_bert.named_parameters())
    bert_list = list(bert.named_parameters())
    # This is the reordering step
    bert_list = [bert_list[-5]] + bert_list[:-5] + bert_list[-4:]
    
    # Initialise an empty dictionary to store the correct key-value pairs
    state_dict = {}

    # Check the number of params/buffers is correct
    assert len(my_bert_list) == len(bert_list), "Number of layers is wrong."
    
    for (my_param_name, my_param), (name, param) in zip(my_bert_list, bert_list):
        state_dict[my_param_name] = param

    if set(state_dict.keys()) != set(my_bert.state_dict().keys()):
        raise Exception("State dicts don't match.")
    
    my_bert.load_state_dict(state_dict)
    
    return my_bert

#%%

if MAIN:
    
    my_bert = copy_weights_from_bert(my_bert, bert)

# %%

def predict(model, tokenizer, text: str, k=15) -> List[List[str]]:
    '''
    Return a list of k strings for each [MASK] in the input.
    '''
    
    model.eval()

    # Get input ids, and generate output
    input_ids = tokenizer.encode(text=text, return_tensors="pt")
    output_logits = model(input_ids)

    if not isinstance(output_logits, t.Tensor):
        output_logits = output_logits.logits
    
    # Iterate through the input_ids, and add predictions for each masked token
    mask_predictions = []
    for i, input_id in enumerate(input_ids.squeeze()):
        if input_id == tokenizer.mask_token_id:
            logits = output_logits[0, i]
            top_logits_indices = t.topk(logits, k).indices
            predictions = tokenizer.decode(top_logits_indices)
            mask_predictions.append(predictions)
    
    return mask_predictions

def test_bert_prediction(predict, model, tokenizer):
    '''Your Bert should know some names of American presidents.'''
    text = "Former President of the United States of America, George[MASK][MASK]"
    predictions = predict(model, tokenizer, text)
    print(f"Prompt: {text}")
    print("Model predicted: \n", "\n".join(map(str, predictions)))
    assert "Washington" in predictions[0]
    assert "Bush" in predictions[0]

#%%

if MAIN:
    
    test_bert_prediction(predict, my_bert, tokenizer)

    your_text = "The Answer to the Ultimate Question of Life, The Universe, and Everything is [MASK]."
    predictions = predict(my_bert, tokenizer, your_text)
    print("Model predicted: \n", "\n".join(map(str, predictions)))


# %%
