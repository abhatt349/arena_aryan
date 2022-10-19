#%%
import numpy as np
from fancy_einsum import einsum
from einops import reduce, rearrange, repeat
from typing import Union, Optional, Callable
import torch as t
import torchvision
import utils
from utils import display_array_as_img

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
    cubeA = matA.as_strided(size=(i,j,k), stride=(sA[0], sA[1], 0))
    cubeB = matB.as_strided(size=(i,j,k), stride=(0, sB[0], sB[1]))

    return einsum('i j k -> i k', cubeA*cubeB)

utils.test_mm(as_strided_mm)
utils.test_mm2(as_strided_mm)
# %%
