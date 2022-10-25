# %%
from cmath import inf
from tracemalloc import start
import torch as t
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import PIL
from PIL import Image
import json
from pathlib import Path
from typing import Union, Tuple, Callable, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import utils
from tqdm.notebook import tqdm_notebook
import time

from w0d2_solutions import *
# %%

class ConvNet(nn.Module): # type: ignore
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.flat1 = Flatten(start_dim=1, end_dim=-1)
        self.fc1 = Linear(in_features=3136, out_features=128, bias=True)
        self.fc2 = Linear(in_features=128, out_features=10, bias=True)


    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flat1(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

model = ConvNet()
# print(model)

# %%

device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

#%%

epochs = 3
loss_fn = nn.CrossEntropyLoss()   # type: ignore
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
testloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)


MODEL_FILENAME = "./w1d2_convnet_mnist.pt"
device = "cuda" if t.cuda.is_available() else "cpu"


def train_convnet(trainloader: DataLoader, testloader: DataLoader, epochs: int, loss_fn: Callable) -> list:
    '''
    Defines a ConvNet using our previous code, and trains it on the data in trainloader.

    Returns tuple of (loss_list, accuracy_list), where accuracy_list contains the fraction of accurate classifications on the test set, at the end of each epoch.
    '''

    model = ConvNet().to(device).train()
    optimizer = t.optim.Adam(model.parameters())
    loss_list = []
    accuracy_list = []

    for epoch in tqdm_notebook(range(epochs)):

        for (x, y) in tqdm_notebook(trainloader, leave=False):

            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.item())
        
        for (x, y) in tqdm_notebook(testloader, leave=False):

            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            preds = t.argmax(y_hat, dim=1)
            accuracy = (y == preds).to(float).mean().item()
            accuracy_list.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}") # type: ignore

    print(f"Saving model to: {MODEL_FILENAME}")
    t.save(model, MODEL_FILENAME)
    return [loss_list, accuracy_list]


loss_list, accuracy_list = train_convnet(trainloader, testloader, epochs, loss_fn)

utils.plot_loss_and_accuracy(loss_list, accuracy_list)


# %%
