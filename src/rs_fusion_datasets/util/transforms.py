from functools import partial
from typing import Callable, Dict, List
from jaxtyping import Float
from numpy import ndarray
import numpy as np
from scipy.sparse import spmatrix

Preprocess = Callable[ [Float[ndarray, 'C H W']], Float[ndarray, 'C H W']]

def Compose(preprocesses: List[Preprocess]) -> Preprocess:
    """
    Compose multiple preprocessing functions into a single function.
    :param preprocesses: List of preprocessing functions to compose.
    :return: A single preprocessing function that applies all the given functions in sequence.
    """
    def composed(x: Float[ndarray, 'C H W']) -> Float[ndarray, 'C H W']:
        for preprocess in preprocesses:
            x = preprocess(x)
        return x
    return composed

def identity(x):
    return x
def Identify() -> Preprocess:
    """
    Identity preprocessing function.
    :return: Identity function that returns the input image unchanged.
    """
    return identity

def map_lbl(lbl: Float[spmatrix, 'H W'], mapping: Dict[int, int]) -> Float[ndarray, 'H W']:
    """
    Map labels in the input label array according to the provided mapping.
    :param lbl: Input label array with shape (H, W).
    :param mapping: Dictionary mapping old labels to new labels.
    :return: Mapped label array with shape (H, W).
    """
    lbl = lbl.copy()
    for old_label, new_label in mapping.items():
        lbl[lbl == old_label] = new_label
    return lbl  # Ensure the output is of type int16
    
def MapLbl(mapping: Dict[int, int]) -> Preprocess:
    """
    Create a preprocessing function that maps labels in the input label array according to the provided mapping.
    :param mapping: Dictionary mapping old labels to new labels.
    :return: Preprocessing function that maps labels.
    """
    return partial(map_lbl, mapping=mapping)


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