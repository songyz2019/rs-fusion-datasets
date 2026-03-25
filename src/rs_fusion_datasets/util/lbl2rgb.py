from colorsys import hsv_to_rgb
from typing import Union
import scipy
import scipy.sparse
import skimage
import torch
import numpy as np
from jaxtyping import UInt8, Float
import warnings

def hue_platte(n_sample :int):
    return [hsv_to_rgb(n/n_sample, 1, 1) for n in range(n_sample)]


def hex2rgb(x :str):
    if not x.startswith("#"):
        return x
    x = x.removeprefix("#")
    r = int(x[:2],  base=16) / 255.0
    g = int(x[2:4], base=16) / 255.0
    b = int(x[4:6], base=16) / 255.0
    return [r,g,b]


def lbl2rgb(lbl :Float[Union[np.ndarray,torch.Tensor,scipy.sparse.spmatrix], 'C H W'], palette :Union[tuple, str]='default') -> UInt8[Union[np.ndarray,torch.Tensor], '... 3 H W']:
    """
    Convert a label image to a RGB image.

    :param lbl a hw label image, or a chw onehot label image with c=n_class. tensor or numpy array or scipy.sparse are supported. An extra batch dimension is allowed. Label should start form 1. 0 means background.
    :param palette: a tuple of hex color strings, or a preset name. The default is a placeholder palette which works for most datasets with less than 32 classes.
    :return: a 3HW RGB image, uint8, the background is black.
    """
    
    if isinstance(lbl, torch.Tensor):
        lbl = lbl.cpu().numpy()
    elif scipy.sparse.issparse(lbl):
        lbl = lbl.todense()
    if len(lbl.shape) == 3:
        lbl = np.argmax(lbl, axis=-3)
    lbl = lbl.astype(np.int16)
    img = skimage.color.label2rgb(
        lbl,
        channel_axis=-3,
        colors=[hex2rgb(x) for x in palette],
        bg_label=0,
        bg_color=hex2rgb('#000000'),
        kind='overlay' # 硬分割还是软分割
    )
    img_u8 = (img * 255).astype(np.uint8)
    return img_u8