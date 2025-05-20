from jaxtyping import Float, Num
from typing import Union, List
import numpy as np
from scipy.sparse import coo_array

# There's a powery third-party library called `patchify` that can be used to patchify images, but we won't use it because we want to have less dependencies

def patchify(imgs :Union[List[Float[np.ndarray, 'c h w']], Float[np.ndarray, 'c h w']], lbl :Num[coo_array, 'h w'], patch_size :int, dtype = np.float32) -> tuple[Union[List[Float[np.ndarray, 'n c h w']], Float[np.ndarray, 'n c h w']], Float[np.ndarray, 'n d']]:
    patch_radius = patch_size // 2
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    n_sample = len(lbl.data)
    n_modal = len(imgs)
    x = [ None for _ in range(n_modal) ] # n_modal, n_patch, n_channel, patch_size, patch_size
    for i_modal in range(n_modal):
        img = np.pad(imgs[i_modal], ((0, 0), (patch_radius, patch_radius), (patch_radius, patch_radius)), 'reflect')
        x[i_modal] = np.zeros((n_sample, img.shape[0], patch_size, patch_size), dtype=dtype)
        for i_sample in range(n_sample):
            r, c = lbl.row[i_sample], lbl.col[i_sample]
            x[i_modal][i_sample, :, :, :] = img[:, r:r+patch_size, c:c+patch_size]
    if len(x) == 1:
        x = x[0]
    return x, lbl.data
