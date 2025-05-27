from functools import partial
from typing import Callable
from jaxtyping import Float
from numpy import ndarray
import numpy as np


Preprocess = Callable[ [Float[ndarray, 'C H W']], Float[ndarray, 'C H W']]


def normalize_per_channel(x: Float[ndarray, 'c h w']) -> Float[ndarray, 'c h w']:
    """
    Normalize each channel of the input image to the range [0, 1].
    :param x: Input image with shape (c, h, w).
    :return: Normalized image with shape (c, h, w).
    """
    x = x.astype(np.float32)
    min_val = np.min(x, axis=(-1, -2), keepdims=True)
    max_val = np.max(x, axis=(-1, -2), keepdims=True)
    scale = max_val - min_val
    scale[scale == 0.0] = 1e-6
    return (x - min_val) / (max_val - min_val)
def NormalizePerChannel() -> Preprocess:
    return normalize_per_channel

def normalize(x: Float[ndarray, 'c h w']) -> Float[ndarray, 'c h w']:
    """
    Normalize the input image to the range [0, 1].
    :param x: Input image with shape (c, h, w).
    :return: Normalized image with shape (c, h, w).
    """
    x = x.astype(np.float32)
    min_val = np.min(x, keepdims=True)
    max_val = np.max(x, keepdims=True)
    if min_val == max_val:
        return x - min_val
    return (x - min_val) / (max_val - min_val)

def Normalize() -> Preprocess:
    """
    Normalize the input image to the range [0, 1].
    :param x: Input image with shape (c, h, w).
    :return: Normalized image with shape (c, h, w).
    """
    return normalize

def channel_pca(x: Float[ndarray, 'c h w'], n_components:int) -> Float[ndarray, 'c h w']:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    c,h,w = x.shape
    x = x.transpose(1, 2, 0).reshape(-1, c)  # Reshape to (h*w, c)
    x = pca.fit_transform(x)
    return x.reshape(h, w, n_components).transpose(2,0,1)  # Reshape back to (n_components, h, w)

def ChannelPCA(n_components: int) -> Preprocess:
    """
    PCA preprocessing function.
    :param n_components: Number of components to keep.
    :return: PCA function.
    """
    return partial(channel_pca, n_components=n_components)