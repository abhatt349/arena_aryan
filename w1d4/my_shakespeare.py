# %%
from ast import Mult
import matplotlib.pyplot as plt
from regex import D
import torch as t
from torch import nn, norm, optim
from torch import Tensor
from torch.nn import functional
from einops import reduce, repeat, rearrange
import plotly.express as px
import numpy as np
from fancy_einsum import einsum
from math import sqrt
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Union, Optional
from tqdm.notebook import tqdm_notebook
from collections import OrderedDict
import transformers
from torch.distributions.categorical import Categorical
import re
from final_transformer import TransformerConfig, DecoderOnlyTransformer, plot_loss_and_accuracy
from w1d4_answers import sample_tokens

# %%

dtype = t.double
device = 'cuda' if t.cuda.is_available() else 'cpu'

# %%

with open("100-0.txt") as file:
    text = file.read()
    words = re.split(r"\b", text)

# %%

class WordsDataset(Dataset):
    def __init__(self, words, seq_len, sample_size):
        
        self.words = words
        self.seq_len = seq_len
        self.sample_size = sample_size
        self.vocab_size = len(set(self.words))
        self.max_len = len(self.words) - self.seq_len + 1
        self.word_to_tok = {word: i for (i, word) in enumerate(set(words))}
        self.tok_to_word = {self.word_to_tok[word]: word for word in self.word_to_tok}
        self.tokens = t.tensor([self.word_to_tok[word] for word in self.words], dtype=t.long, device=device)

    def __len__(self):
        return int(self.max_len * self.sample_size)

    def __getitem__(self, idx):

        current_seq = self.tokens[idx: idx + self.seq_len + 1]
        x = current_seq[:-1]
        y = current_seq[1:]

        return x, y


# %%

class WordsTokenizer():
    model_max_length: int

    def __init__(self, wordsdataset: WordsDataset):
        
        self.word_to_tok = wordsdataset.word_to_tok
        self.tok_to_word = wordsdataset.tok_to_word
        self.model_max_length = wordsdataset.seq_len

    def encode(self, initial_text: str, return_tensors: Optional[str] = None) -> Union[list, t.Tensor]:
        '''
        Tokenizes initial_text, then returns the token ids.

        Return type is list by default, but if return_tensors="pt" then it is returned as a tensor.
        '''

        words = re.split(r'\b', initial_text)
        words = [word for word in words if word]

        tokens = [self.word_to_tok[word] for word in words]
        if return_tensors == 'pt':
            tokens = t.tensor(tokens, dtype=dtype, device=device)

        return tokens
        

    def decode(self, list_of_ids: Union[t.Tensor, list]) -> str:
        '''
        Converts ids to a list of tokens, then joins them into a single string.
        '''
        
        return ''.join([self.tok_to_word[int(token)] for token in list_of_ids])


# %%

def train_transformer(model, loss_fn, optimizer, trainloader, epochs, plot_loss=True):

    loss_list = []

    for epoch in range(epochs):
        
        progress_bar = tqdm_notebook(trainloader)
        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            y_hat = rearrange(model(x), "b s d -> (b s) d")
            y = t.flatten(y)

            loss = loss_fn(y_hat, y)
            loss_list.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.set_description(f"epoch = {epoch+1}, loss = {loss.item():.4f}")

    # Function to plot the loss over epochs
    if plot_loss:
        fig = px.line(
            y=loss_list, 
            template="simple_white", 
            labels={
                "x": "No. batches seen", 
                "y": str(loss_fn).replace("()", "") # This gets a name like "CrossEntropyLoss" from the loss function
            }, 
            title='Training loss'
        )
        # This next bit of code plots vertical lines corresponding to the epochs
        if epochs > 1:
            for idx, epoch_start in enumerate(np.linspace(0, len(loss_list), epochs, endpoint=False)):
                fig.add_vline(x=epoch_start, line_width=3, line_dash="dash", annotation_text=f"Epoch {idx}", annotation_position="top right")
        fig.show()
    
    return model

# %%

max_seq_len = 48
batch_size = 32

trainset = WordsDataset(words=words, seq_len=max_seq_len, sample_size=0.00002)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=t.cuda.is_available())
tokenizer = WordsTokenizer(trainset)

# %%

config = TransformerConfig(
    num_layers = 8,
    num_heads = 8,
    vocab_size = trainset.vocab_size,
    hidden_size = 512,
    max_seq_len = trainset.seq_len,
    dropout = 0.1,
    layer_norm_epsilon = 1e-05
)

model = DecoderOnlyTransformer(config).to(device).train()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 1

# %%

model = train_transformer(model, loss_fn, optimizer, trainloader, epochs)

# %%

initial_text = "twas"
text_output = sample_tokens(model, tokenizer, initial_text, max_tokens_generated=100, temperature=1.0, top_k=10)
print(text_output)

# %%
