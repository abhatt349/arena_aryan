# %%
import torch as t
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fancy_einsum import einsum
from typing import Union, Optional, Callable
import numpy as np
from einops import rearrange
from tqdm.notebook import tqdm_notebook
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import wandb
import utils
from w0d3_solutions import ResNet34

MAIN = (__name__ == '__main__')
device = "cuda" if t.cuda.is_available() else "cpu"

# %%

if MAIN:
    cifar_mean = [0.485, 0.456, 0.406]
    cifar_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_mean, std=cifar_std)
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    utils.show_cifar_images(trainset, rows=3, cols=5)

# %%

def train(trainset, testset, epochs: int, loss_fn: Callable, batch_size: int, lr: float) -> tuple[list, list]:

    model = ResNet34().to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    loss_list = []
    accuracy_list = []

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    for epoch in range(epochs):

        progress_bar = tqdm_notebook(trainloader)

        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

        with t.inference_mode():

            accuracy = 0
            total = 0

            for (x, y) in testloader:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            accuracy_list.append(accuracy/total)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {accuracy}/{total}")

    filename = "./w0d3_resnet.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)

    utils.plot_results(loss_list, accuracy_list)
    return loss_list, accuracy_list

if MAIN:
    epochs = 1
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 128
    lr = 0.001

    loss_list, accuracy_list = train(trainset, testset, epochs, loss_fn, batch_size, lr)

# %%

def train(trainset, testset, epochs: int, loss_fn: Callable, batch_size: int, lr: float) -> None:

    config_dict = {
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
    }
    wandb.init(project="w2d1_resnet", config=config_dict)

    model = ResNet34().to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    examples_seen = 0
    start_time = time.time()

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(epochs):

        progress_bar = tqdm_notebook(trainloader)

        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

            examples_seen += len(x)
            wandb.log({"train_loss": loss, "elapsed": time.time() - start_time}, step=examples_seen)

        with t.inference_mode():

            accuracy = 0
            total = 0

            for (x, y) in testloader:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)

    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)

if MAIN:
    train(trainset, testset, epochs, loss_fn, batch_size, lr)