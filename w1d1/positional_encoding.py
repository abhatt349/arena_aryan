# %%
import matplotlib.pyplot as plt
import seaborn as sns
import torch as t
from einops import reduce, repeat, rearrange
import plotly.express as px
import numpy as np

# %%

L = 32
d = 128

def PE(L, d):
    # graph1 = repeat(t.arange(L), 'L -> L d', d=d//2)
    # graph2 = 1 / 1e4 ** repeat(t.arange(0,d,step=2)/d, 'd -> L d', L=L)

    # graph = graph1 * graph2

    graph1 = t.arange(L)
    graph2 = 1 / 1e4 ** (t.arange(0,d,step=2)/d)

    graph = t.outer(graph1, graph2)
    graph = rearrange(t.cat([t.sin(graph), t.cos(graph)], dim=1), 'L (d1 d2) -> L (d2 d1)', d1=2)

    return graph

# %%

graph = PE(L, d)

# sns.heatmap(graph)
# plt.show()

px.imshow(graph, color_continuous_scale='RdBu')

# %%

# My method
px.imshow(graph @ graph.T / (d//2), color_continuous_scale='Blues')

# %%

# Given method

def cosine_similarity(vec1, vec2):

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_dot_product_graph(array_2d):

    (L, d) = array_2d.shape

    arr = np.zeros((L, L))

    # Note, there are also more elegant ways to do this than with a for loop!
    for i in range(L):
        for j in range(L):
            arr[i, j] = cosine_similarity(array_2d[i], array_2d[j])

    px.imshow(arr, color_continuous_scale="Blues").show()
    return arr

# %%

# Ensure that my method gives the same results as solution
print((t.tensor(get_dot_product_graph(graph)) - (graph @ graph.T / (d//2))).abs().max())
assert t.allclose(t.tensor(get_dot_product_graph(graph), dtype=t.float32), graph @ graph.T / 64)

# %%
