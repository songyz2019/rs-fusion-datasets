# Assume that torch exists
from typing import Callable, Literal, Optional

from scipy.sparse import coo_array
from torchvision.datasets.vision import VisionDataset
import torch
import numpy as np
import warnings

from ..core.common import DataMetaInfo

class CommonHsiDsmDataset(VisionDataset):
    def __init__(self,
                 data_fetch: Callable[[],tuple[np.ndarray,np.ndarray,coo_array,coo_array, DataMetaInfo]],
                 subset: Literal['train', 'test', 'full'], 
                 patch_size: int = 11,  # I prefer patch_radius, but patch_size is more popular and maintance two pathc_xxx is to complex...
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subset = subset

        if patch_size % 2 != 1:
            pass
            # warnings.warn("Non-odd patch size may cause unknown behaviors (classification pixel is at right bottom 2x2 center location). Use at your own risk.",category=UserWarning,stacklevel=2)
        self.patch_size = patch_size
        self.patch_radius = patch_size // 2 # if patch_size is odd, it should be (patch_size - w_center)/2, but some user will use on-odd patch size

        self.HSI, self.LIDAR, train_truth, test_truth, self.INFO = data_fetch()
        self.n_class = self.INFO['n_class']
        self.n_channel_hsi   = self.INFO['n_channel_hsi']
        self.n_channel_lidar = self.INFO['n_channel_lidar']


        # Preprocess HSI and DSM
        pad_shape = ((0, 0), (self.patch_radius, self.patch_radius), (self.patch_radius, self.patch_radius))

        self.hsi = self.HSI
        self.dsm = self.LIDAR
        self.dsm = (self.dsm - self.dsm.min()) / (self.dsm.max() - self.dsm.min())

        # If we have einops, just need
        # min_value = reduce(x, '... c h w -> ... c 1 1', 'min')
        # max_value = reduce(x, '... c h w -> ... c 1 1', 'max')
        # return (x-min_value) / (max_value-min_value)
        min_hsi = self.hsi.min(axis=-1, keepdims=True).min(axis=-2, keepdims=True)
        max_hsi = self.hsi.max(axis=-1, keepdims=True).max(axis=-2, keepdims=True)
        self.hsi = (self.hsi - min_hsi) / (max_hsi - min_hsi)
        
        self.hsi = np.pad(self.hsi,   pad_shape, 'symmetric')
        self.dsm = np.pad(self.dsm, pad_shape, 'symmetric')

        # Preprocess truth
        self.truth = {
            'train': train_truth,
            'test': test_truth,
            'full': coo_array(-1*np.ones_like(train_truth.todense(), dtype=np.int16), dtype='int')
        }[subset]

        self.hsi = torch.from_numpy(self.hsi).float()
        self.dsm = torch.from_numpy(self.dsm).float()


    def __len__(self):
        return self.truth.count_nonzero()

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        w = self.patch_size

        i = self.truth.row[index]
        j = self.truth.col[index]
        cid = self.truth.data[index].item()

        x_hsi = self.hsi[:, i:i+w, j:j+w]
        x_dsm = self.dsm[:, i:i+w, j:j+w]
        y = np.eye(self.n_class)[cid-1]
        y = torch.from_numpy(y).float()

        # 额外信息: 当前点的坐标
        extras = {
            "location": [i, j],
            "index": index,
            "class": cid
        }

        return x_hsi, x_dsm, y, extras
