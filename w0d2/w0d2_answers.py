#%%
from xml.dom.pulldom import START_DOCUMENT
import numpy as np
from fancy_einsum import einsum
from einops import reduce, rearrange, repeat
from typing import Union, Optional, Callable
import torch as t
from torch import nn
import torchvision
import utils
from utils import display_array_as_img
import functools
import math

arr = np.load("numbers.npy")

#%%
display_array_as_img(rearrange(arr, 'b c h w -> c h (b w)'))

# %%
display_array_as_img(repeat(arr[0,...], 'c h w -> c (2 h) w',))

# %%
display_array_as_img(repeat(arr[0:2,...], 'b c h w -> c (b h) (2 w)',))

# %%
display_array_as_img(repeat(arr[0,...], 'c h w -> c (h 2) w',))

# %%
display_array_as_img(rearrange(arr[0], "c h w -> h (c w)"))

# %%
display_array_as_img(rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2))

# %%
display_array_as_img(reduce(arr, "b c h w -> h (b w)", 'max'))

# %%
display_array_as_img(reduce(arr.astype(float), "b c h w -> h (b w)", 'mean').astype(int))

# %%
display_array_as_img(reduce(arr, "b c h w -> h w", 'min'))

# %%
display_array_as_img(rearrange(arr[0:2,...], "b c (h1 h2) w -> c h2 (h1 b w)", h1=2))

# %%
display_array_as_img(rearrange(arr[1,...], "c h w -> c w h"))

# %%
display_array_as_img(rearrange(arr, "(b1 b2) c h w -> c (b1 w) (b2 h)", b1=2))

# %%
display_array_as_img(reduce(arr.astype(float), "(b1 b2) c (h 2) (w 2) -> c (b1 h) (b2 w)", 'mean', b1=2).astype(int))
display_array_as_img(reduce(arr, "(b1 b2) c (h 2) (w 2) -> c (b1 h) (b2 w)", 'max', b1=2))

# %%
display_array_as_img(repeat(reduce(arr, 'b c h w -> b c', 'max'), '(b1 b2) c -> c (b1 150) (b2 150)', b1=2))
# %%

def einsum_trace(mat: np.ndarray):
    """
    Returns the same as `np.trace`.
    """
    return einsum('i i', mat)

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    """
    return einsum('i j, j -> i', mat, vec)

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    """
    return einsum('i j, j k -> i k', mat1, mat2)

def einsum_inner(vec1, vec2):
    """
    Returns the same as `np.inner`.
    """
    return einsum('i,i', vec1, vec2)

def einsum_outer(vec1, vec2):
    """
    Returns the same as `np.outer`.
    """
    return einsum('i,j -> i j', vec1, vec2)

utils.test_einsum_trace(einsum_trace)
utils.test_einsum_mv(einsum_mv)
utils.test_einsum_mm(einsum_mm)
utils.test_einsum_inner(einsum_inner)
utils.test_einsum_outer(einsum_outer)
# %%

test_input = t.tensor(
    [[0, 1, 2, 3, 4], 
    [5, 6, 7, 8, 9], 
    [10, 11, 12, 13, 14], 
    [15, 16, 17, 18, 19]], dtype=t.float
)

# %%

import torch as t
from collections import namedtuple

TestCase = namedtuple("TestCase", ["output", "size", "stride"])

test_cases = [
    TestCase(
        output=t.tensor([0, 1, 2, 3]), 
        size=(4,), 
        stride=(1,)),
    TestCase(
        output=t.tensor([0, 1, 2, 3, 4]), 
        size=(5,), 
        stride=(1,)),
    TestCase(
        output=t.tensor([0, 5, 10, 15]), 
        size=(4,), 
        stride=(5,)),
    TestCase(
        output=t.tensor([[0, 1, 2], [5, 6, 7]]), 
        size=(2,3), 
        stride=(5,1)),
    TestCase(
        output=t.tensor([[0, 1, 2], [10, 11, 12]]), 
        size=(2,3), 
        stride=(10,1)),
    TestCase(
        output=t.tensor([[0, 0, 0], [11, 11, 11]]), 
        size=(2,3),
        stride=(11,0)),    
    TestCase(
        output=t.tensor([0, 6, 12, 18]), 
        size=(4,), 
        stride=(6,)),
    TestCase(
        output=t.tensor([[[0, 1, 2]], [[9, 10, 11]]]), 
        size=(2,1,3), 
        stride=(9,0,1)),
    TestCase(
        output=t.tensor([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[12, 13], [14, 15]], [[16, 17], [18, 19]]]]),
        size=(2,2,2,2),
        stride=(12,4,2,1)),]
for (i, case) in enumerate(test_cases):
    if (case.size is None) or (case.stride is None):
        print(f"Test {i} failed: attempt missing.")
    else:
        actual = test_input.as_strided(size=case.size, stride=case.stride)
        if (case.output != actual).any():
            print(f"Test {i} failed:")
            print(f"Expected: {case.output}")
            print(f"Actual: {actual}")
        else:
            print(f"Test {i} passed!")
# %%

def as_strided_trace(mat: t.Tensor) -> t.Tensor:
    '''
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    '''

    return mat.as_strided(size=(mat.shape[0],), stride=(sum(mat.stride()),)).sum()

utils.test_trace(as_strided_trace)
# %%

def as_strided_mv(mat: t.Tensor, vec: t.Tensor) -> t.Tensor:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    return einsum('i j -> i', mat*vec.as_strided(size=mat.shape, stride=(0,vec.stride()[0])))

utils.test_mv(as_strided_mv)
utils.test_mv2(as_strided_mv)
# %%

def as_strided_mm(matA: t.Tensor, matB: t.Tensor) -> t.Tensor:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    sA = matA.stride()
    sB = matB.stride()
    i,j,k = matA.shape[0], matA.shape[1], matB.shape[1]
    cubeA = matA.as_strided(size=(i,j,k), stride=(sA[0], sA[1], 0))  # type: ignore
    cubeB = matB.as_strided(size=(i,j,k), stride=(0, sB[0], sB[1]))  # type: ignore

    return einsum('i j k -> i k', cubeA*cubeB)

utils.test_mm(as_strided_mm)
utils.test_mm2(as_strided_mm)
# %%

def conv1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''

    batch = x.shape[0]
    in_channels = x.shape[1]
    width = x.shape[2]
    out_channels = weights.shape[0]
    kernel_width = weights.shape[2]
    out_width = width - kernel_width + 1

    x_batch_stride, x_in_c_stride, x_width_stride = x.stride()  # type: ignore

    x_strided = x.as_strided(size=(batch, in_channels, out_width, kernel_width), 
                            stride=(x_batch_stride, x_in_c_stride, x_width_stride, x_width_stride))  # type: ignore
    return einsum('b in_c out_w kernel_w, out_c in_c kernel_w -> b out_c out_w', x_strided, weights)

utils.test_conv1d_minimal(conv1d_minimal)
# %%

def conv2d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    
    batch = x.shape[0]
    in_channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    out_channels = weights.shape[0]
    kernel_height = weights.shape[2]
    kernel_width = weights.shape[3]

    out_height = height - kernel_height + 1
    out_width = width - kernel_width + 1

    xs = x.stride()

    x_strided = x.as_strided(size=(batch, in_channels, out_height, out_width, kernel_height, kernel_width), stride=(xs[0], xs[1], xs[2], xs[3], xs[2], xs[3]))   # type: ignore
    return einsum('b in_c out_h out_w ker_h ker_w, out_c in_c ker_h ker_w -> b out_c out_h out_w', x_strided, weights)
    

utils.test_conv2d_minimal(conv2d_minimal)
# %%

def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    padded = x.new_full(size=(x.shape[0], x.shape[1], x.shape[2]+left+right), fill_value=pad_value)

    padded[...,left:left+x.shape[2]] = x 

    return padded


utils.test_pad1d(pad1d)
utils.test_pad1d_multi_channel(pad1d)

# %%

def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    padded = x.new_full(size=(x.shape[0], x.shape[1], x.shape[2]+top+bottom,  x.shape[3]+left+right), fill_value=pad_value)

    padded[...,top:top+x.shape[2],left:left+x.shape[3]] = x 

    return padded

utils.test_pad2d(pad2d)
utils.test_pad2d_multi_channel(pad2d)
# %%

def conv1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    '''Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    
    batch, in_channels, width = x.shape
    out_channels, in_channels, kernel_width = weights.shape

    out_width = ((width + 2 * padding - kernel_width) // stride) + 1
    x = pad1d(x, padding, padding, 0)

    bS, icS, wS = x.stride()  # type: ignore

    new_size = (batch, in_channels, out_width, kernel_width)
    new_stride = (bS, icS, stride * wS, wS)

    x_strided = x.as_strided(size=new_size, stride=new_stride)
    return einsum('batch in_channels out_width kernel_width, out_channels in_channels kernel_width -> batch out_channels out_width', x_strided, weights) 

utils.test_conv1d(conv1d)
# %%

IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

# Examples of how this function can be used:
#       force_pair((1, 2))     ->  (1, 2)
#       force_pair(2)          ->  (2, 2)
#       force_pair((1, 2, 3))  ->  ValueError
# %%

def conv2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    '''Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    batch, in_channels, height, width = x.shape
    out_channels, in_channels, kernel_height, kernel_width = weights.shape

    padding = force_pair(padding)
    stride = force_pair(stride)

    out_height = ((height + 2 * padding[0] - kernel_height) // stride[0]) + 1
    out_width = ((width + 2 * padding[1] - kernel_width) // stride[1]) + 1

    x = pad2d(x, left=padding[1], right=padding[1], top=padding[0], bottom=padding[0], pad_value=0)

    bS, icS, hS, wS = x.stride()  # type: ignore

    new_size = (batch, in_channels, out_height, kernel_height, out_width, kernel_width)
    new_stride = (bS, icS, stride[0] * hS, hS, stride[1] * wS, wS)

    x_strided = x.as_strided(size=new_size, stride=new_stride)
    return einsum('batch in_channels out_height kernel_height out_width kernel_width, out_channels in_channels kernel_height kernel_width -> batch out_channels out_height out_width', x_strided, weights) 

utils.test_conv2d(conv2d)
# %%

def maxpool2d(x: t.Tensor, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 0) -> t.Tensor:
    '''Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, out_height, output_width)
    '''
    kernel_size = force_pair(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = force_pair(stride)
    padding = force_pair(padding)

    (batch, channels, height, width) = x.shape

    out_height = ((height + 2 * padding[0] - kernel_size[0]) // stride[0]) + 1
    out_width = ((width + 2 * padding[1] - kernel_size[1]) // stride[1]) + 1

    x = pad2d(x, left=padding[1], right=padding[1], top=padding[0], bottom=padding[0], pad_value=-float('inf'))

    bS, cS, hS, wS = x.stride()  # type: ignore

    new_size = (batch, channels, out_height, kernel_size[0], out_width, kernel_size[1])
    new_stride = (bS, cS, stride[0] * hS, hS, stride[1] * wS, wS)

    x_strided = x.as_strided(size=new_size, stride=new_stride)

    return reduce(x_strided, 'batch channels out_height kernel_height out_width kernel_width -> batch channels out_height out_width', 'max')


utils.test_maxpool2d(maxpool2d)
# %%

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):

        super().__init__()

        self.kernel_size = force_pair(kernel_size)
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = force_pair(stride)
        self.padding = force_pair(padding)


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of maxpool2d.'''
        return maxpool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        return f"MaxPool layer with kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"

utils.test_maxpool2d_module(MaxPool2d)
m = MaxPool2d(kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")
# %%

class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(x, t.zeros(x.shape))

utils.test_relu(ReLU)

# %%
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim=start_dim
        self.end_dim=end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''

        start_dim = self.start_dim % len(input.shape)
        end_dim = self.end_dim % len(input.shape) + 1

        new_shape = list(input.shape)
        new_dim = functools.reduce(lambda x,y:x*y, new_shape[start_dim:end_dim], 1)
        new_shape = new_shape[:self.start_dim] + [new_dim] + new_shape[end_dim:]
        
        return t.reshape(input, shape=tuple(new_shape))


    def extra_repr(self) -> str:
        return f"Flatten layer with start_dim={self.start_dim}, end_dim={self.end_dim}"

utils.test_flatten(Flatten)
# %%

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        norm = math.sqrt(1/in_features)

        weights = t.rand(size=(out_features, in_features), dtype=t.float)
        weights = (2 * weights - 1) * norm

        self.weight = nn.Parameter(weights)

        if bias:
            self.bias = nn.Parameter((2*t.rand(size=(out_features,), dtype=t.float)-1)*norm)
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        output = einsum('out_features in_features, ... in_features -> ... out_features', self.weight, x)

        if self.bias is not None:
            output += self.bias

        return output


    def extra_repr(self) -> str:
        return f"Linear layer with in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, bias={self.bias is not None}"

utils.test_linear_forward(Linear)
utils.test_linear_parameters(Linear)
utils.test_linear_no_bias(Linear)
# %%

class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = force_pair(kernel_size)
        self.stride = force_pair(stride)
        self.padding = force_pair(padding)

        norm = math.sqrt(1/(in_channels*self.kernel_size[0]*self.kernel_size[1]))

        weight = t.rand(size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]), dtype=t.float)
        weight = (2 * weight - 1) * norm

        self.weight = nn.Parameter(weight)



    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the functional conv2d you wrote earlier.'''
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        return f"Conv2d layer with in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.weight.shape}, stride={self.stride}, padding={self.padding}"

utils.test_conv2d_module(Conv2d)
# %%
