from functools import partial
from typing import Callable, Dict, List
from jaxtyping import Float
from numpy import ndarray
import numpy as np
from scipy.sparse import spmatrix

Preprocess = Callable[ [Float[ndarray, 'C H W']], Float[ndarray, 'C H W']]
LblPreprocess = Callable[ [Float[spmatrix, 'C H W']], Float[spmatrix, 'C H W']]

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

def map_lbl(lbl: Float[spmatrix, 'H W'], mapping: Dict[int, int]) -> Float[spmatrix, 'H W']:
    """
    Map labels in the input label array according to the provided mapping.
    :param lbl: Input label array with shape (H, W).
    :param mapping: Dictionary mapping old labels to new labels.
    :return: Mapped label array with shape (H, W).
    """
    new_lbl = lbl.toarray().copy() if hasattr(lbl, "toarray") else lbl.copy()
    for old_label, new_label in mapping.items():
        new_lbl[lbl == old_label] = new_label
    return new_lbl
    
def MapLbl(mapping: Dict[int, int]) -> LblPreprocess:
    """
    Create a preprocessing function that maps labels in the input label array according to the provided mapping.
    :param mapping: Dictionary mapping old labels to new labels.
    :return: Preprocessing function that maps labels.
    """
    return partial(map_lbl, mapping=mapping)


def normalize(x: Float[ndarray, 'c h w'], axis=(-1,-2,-3)) -> Float[ndarray, 'c h w']:
    """
    Normalize the input image to the range [0, 1].
    :param x: Input image with shape (c, h, w).
    :return: Normalized image with shape (c, h, w).
    """
    x = x.astype(np.float32)
    min_val = np.min(x, axis=axis, keepdims=True)
    max_val = np.max(x, axis=axis, keepdims=True)
    # This does not work for some edge cases about axis
    # if min_val == max_val:
        # return x - min_val
    result = (x - min_val) / (max_val - min_val)
    result = np.nan_to_num(result)  # convert NaN to 0
    return result

def Normalize(axis=(-1,-2,-3)) -> Preprocess:
    """
    Normalize the input image to the range [0, 1].
    :param x: Input image with shape (c, h, w).
    :return: Normalized image with shape (c, h, w).
    """
    return partial(normalize, axis=axis)

def NormalizePerChannel() -> Preprocess:
    return partial(normalize, axis=(-1, -2))

def standardize(x: Float[ndarray, 'c h w'], axis=(-1,-2,-3)) -> Float[ndarray, 'c h w']:
    """
    Standardize the input image to have zero mean and unit variance.
    :param x: Input image with shape (c, h, w).
    :return: Standardized image with shape (c, h, w).
    """
    x = x.astype(np.float32)
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std[std == 0.0] = 1e-6
    return (x - mean) / std

def Standardize(axis=(-1, -2, -3)) -> Preprocess:
    """
    Standardize the input image to have zero mean and unit variance.
    :return: Standardization function.
    """
    return partial(standardize, axis=axis)

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