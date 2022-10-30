# %%
from tracemalloc import start
from turtle import right
import torch as t
from torch import nn
from torch.nn import Module
from torch.nn.parameter import Parameter
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
from einops import rearrange, reduce, repeat

from w0d2_solutions import *


MAIN = (__name__ == '__main__')
# %%

class ConvNet(Module): 
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
loss_fn = t.nn.CrossEntropyLoss()   
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)


MODEL_FILENAME = "./w0d3_convnet_mnist.pt"
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

        loss = None
        
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

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}") 

    print(f"Saving model to: {MODEL_FILENAME}")
    t.save(model, MODEL_FILENAME)
    return [loss_list, accuracy_list]


# loss_list, accuracy_list = train_convnet(trainloader, testloader, epochs, loss_fn)

# utils.plot_loss_and_accuracy(loss_list, accuracy_list)


# %%

class Sequential(Module):   
    def __init__(self, *modules: Module):   
        super().__init__()
        for i, mod in enumerate(modules):
            self.add_module(str(i), mod)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Chain each module together, with the output from one feeding into the next one.'''
        for mod in self._modules.values():
            if mod is not None: x = mod(x)
        return x


# %%

class BatchNorm2d(Module):   
    running_mean: t.Tensor         # shape: (num_features,)
    running_var: t.Tensor          # shape: (num_features,)
    num_batches_tracked: t.Tensor  # shape: ()

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        weight = t.ones(size=(num_features,), dtype=t.float, device=device)
        bias = t.zeros(size=(num_features,), dtype=t.float, device=device)
        self.weight = Parameter(weight)   
        self.bias = Parameter(bias)   

        running_mean = t.zeros(size=(num_features,), dtype=t.float, device=device)
        running_var = t.ones(size=(num_features,), dtype=t.float, device=device)
        self.register_buffer('running_mean', running_mean)
        self.register_buffer('running_var', running_var)
        self.register_buffer('num_batches_tracked', t.tensor(0))


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        
        if self.training:

            curr_mean = t.mean(x, dim=(0,2,3))
            curr_var = t.var(x, dim=(0,2,3), unbiased=False)

            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*curr_mean
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*curr_var

            self.num_batches_tracked += 1

        else:
            curr_mean = self.running_mean
            curr_var = self.running_var

        curr_mean = repeat(curr_mean, 'c -> b c h w', b=x.shape[0], h=x.shape[2], w=x.shape[3])
        curr_var = repeat(curr_var, 'c -> b c h w', b=x.shape[0], h=x.shape[2], w=x.shape[3])

        normalized_input =  (x-curr_mean)/t.sqrt(curr_var + self.eps)

        weight = repeat(self.weight, 'c -> b c h w', b=x.shape[0], h=x.shape[2], w=x.shape[3])
        bias = repeat(self.bias, 'c -> b c h w', b=x.shape[0], h=x.shape[2], w=x.shape[3])

        return normalized_input * weight + bias


    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["num_features", "eps", "momentum", 'num_batches_tracked']])
        

if MAIN:
    utils.test_batchnorm2d_module(BatchNorm2d)
    utils.test_batchnorm2d_forward(BatchNorm2d)
    utils.test_batchnorm2d_running_mean(BatchNorm2d)
# %%

class AveragePool(Module):   
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return reduce(x, 'b c h w -> b c', 'mean')

# %%

class ResidualBlock(Module):   
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        super().__init__()

        self.left_branch = Sequential(
            Conv2d(in_channels=in_feats, 
                   out_channels=out_feats, 
                   kernel_size=3, 
                   stride=first_stride, 
                   padding=1), 
            BatchNorm2d(num_features=out_feats),
            ReLU(),
            Conv2d(in_channels=out_feats,
                   out_channels=out_feats,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            BatchNorm2d(num_features=out_feats)
        )

        if first_stride == 1:
            self.right_branch = t.nn.Identity()
        else:
            self.right_branch = Sequential(
                Conv2d(in_channels=in_feats,
                       out_channels=out_feats,
                       kernel_size=1,
                       stride=first_stride),
                BatchNorm2d(num_features=out_feats)
            )
        
        self.combine = ReLU()


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''

        left_branch_out = self.left_branch(x)
        right_branch_out = self.right_branch(x)
        return self.combine(left_branch_out + right_branch_out)


# %%

class BlockGroup(Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''

        super().__init__()

        block_list = []

        for i in range(n_blocks):
            if i == 0:
                block_list.append(ResidualBlock(in_feats, out_feats, first_stride))
            else:
                block_list.append(ResidualBlock(out_feats, out_feats, first_stride=1))

        self.res_blocks = Sequential(*block_list)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.
        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        
        return self.res_blocks(x)

# %%

class ResNet34(Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()

        self.beginning = Sequential(
            Conv2d(
                in_channels=3, 
                out_channels=64, 
                kernel_size=7, 
                stride=2, 
                padding=3),
            BatchNorm2d(num_features=64),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        groups = []
        for i in range(len(n_blocks_per_group)):
            groups.append(BlockGroup(
                n_blocks=n_blocks_per_group[i], 
                in_feats=(out_features_per_group[i-1] if (i>0) else 64),
                out_feats=out_features_per_group[i],
                first_stride=first_strides_per_group[i]
                )
            )
        
        self.res_block_groups = Sequential(*groups)
        
        self.ending = Sequential(
            AveragePool(),
            Flatten(),
            Linear(in_features=out_features_per_group[-1], out_features=n_classes)
        )



    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)

        Return: shape (batch, n_classes)
        '''
        
        x = self.beginning(x)
        x = self.res_block_groups(x)
        x = self.ending(x)

        return x

# %%

pretrained_resnet = torchvision.models.resnet34(pretrained=True)

# %%

myresnet = ResNet34()

# %%

def copy_weights(myresnet: ResNet34, pretrained_resnet: torchvision.models.resnet.ResNet) -> ResNet34:
    '''Copy over the weights of `pretrained_resnet` to your resnet.'''

    mydict = myresnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()

    # Check the number of params/buffers is correct
    assert len(mydict) == len(pretraineddict), "Number of layers is wrong. Have you done the prev step correctly?"

    # Initialise an empty dictionary to store the correct key-value pairs
    state_dict_to_load = {}

    for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items()):
        state_dict_to_load[mykey] = pretrainedvalue

    myresnet.load_state_dict(state_dict_to_load)   # type: ignore

    return myresnet

myresnet = copy_weights(myresnet, pretrained_resnet)

# %%

IMAGE_FILENAMES = [
    "chimpanzee.jpg",
    "golden_retriever.jpg",
    "platypus.jpg",
    "frogs.jpg",
    "fireworks.jpg",
    "astronaut.jpg",
    "iguana.jpg",
    "volcano.jpg",
    "goofy.jpg",
    "dragonfly.jpg",
]

IMAGE_FOLDER = Path("./resnet_inputs")

images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]

# %%

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize(size=(224,224))
]) # fill this in as instructed

# %%

def prepare_data(images: list[Image.Image]) -> t.Tensor:
    '''
    Return: shape (batch=len(images), num_channels=3, height=224, width=224)
    '''
    return t.stack([transform(img) for img in images], dim=0)    # type: ignore

prepared_images = prepare_data(images)

# %%

def predict(model, images):
    logits = model(images)
    return logits.argmax(dim=1)

with open("imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())

# %%

my_predictions = predict(myresnet, prepared_images)
pretrained_predictions = predict(pretrained_resnet, prepared_images)

# %%

for i in range(len(images)):
    display(images[i])   # type: ignore
    print('My Prediction:', imagenet_labels[my_predictions[i]])
    print('Real Prediction:', imagenet_labels[pretrained_predictions[i]])
    
# %%
