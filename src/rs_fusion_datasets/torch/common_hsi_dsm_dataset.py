from typing import Literal,Tuple, Dict
import warnings

from scipy.sparse import coo_array
from torchvision.datasets.vision import VisionDataset
import torch
import numpy as np
from numpy import ndarray
from jaxtyping import Num, Float

from ..util.benchmarker import Benchmarker
from ..util.lbl2rgb import lbl2rgb
from ..util.hsi2rgb import hsi2rgb
from ..core.common import DataMetaInfo

def _channel_wise_normalize(x: Float[ndarray, 'c h w']) -> Float[ndarray, 'c h w']:
    """
    Normalize each channel of the input image to the range [0, 1].
    :param x: Input image with shape (c, h, w).
    :return: Normalized image with shape (c, h, w).
    """
    x = x.astype(np.float32)
    min_val = np.min(x, axis=(-1, -2), keepdims=True)
    max_val = np.max(x, axis=(-1, -2), keepdims=True)
    return (x - min_val) / (max_val - min_val)  # Avoid division by zero

class CommonHsiDsmDataset(VisionDataset):
    def __init__(self,
                 hsi :Num[ndarray, 'c h w'], 
                 dsm :Num[ndarray, 'd h w'],
                 lbl_train :Num[ndarray, 'h w'],
                 lbl_test :Num[ndarray, 'h w'],
                 info: DataMetaInfo,
                 split: Literal['train', 'test', 'full'], 
                 patch_size: int = 5,  # I prefer patch_radius, but patch_size is more popular and maintance two patch_xxx is too complex...
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load truth
        self.subset = split
        if self.subset == 'train':
            self.lbl = lbl_train
        elif self.subset == 'test':
            self.lbl = lbl_test
        elif self.subset == 'full':
            self.lbl = coo_array(-1*np.ones_like(lbl_train.todense(), dtype=np.int16), dtype='int')
        else:
            raise ValueError(f"Unknown subset: {split}")

        # Load patch size
        if patch_size % 2 != 1:
            pass
            # warnings.warn("Non-odd patch size may cause unknown behaviors (classification pixel is at right bottom 2x2 center location). Use at your own risk.",category=UserWarning,stacklevel=2)
        self.patch_size = patch_size
        self.patch_radius = patch_size // 2 # if patch_size is odd, it should be (patch_size - w_center)/2, but some user will use on-odd patch size

        # Load basic info
        self.HSI, self.DSM, self.INFO = hsi, dsm, info
        self.n_class = self.INFO['n_class']
        self.n_channel_hsi = self.INFO['n_channel_hsi']
        self.n_channel_dsm = self.INFO['n_channel_dsm']

        # Preprocess HSI
        pad_shape = ((0, 0), (self.patch_radius, self.patch_radius), (self.patch_radius, self.patch_radius))
        self.hsi = _channel_wise_normalize(self.HSI)
        self.hsi = np.pad(self.hsi, pad_shape, mode='reflect')
        self.hsi = torch.from_numpy(self.hsi).float()

        # Preprocess DSM
        self.dsm = _channel_wise_normalize(self.DSM)
        self.dsm = np.pad(self.dsm, pad_shape, mode='reflect')
        self.dsm = torch.from_numpy(self.dsm).float()


    def __len__(self):
        return self.lbl.count_nonzero()

    def __getitem__(self, index) -> Tuple[
            Float[ndarray, 'c h w'],
            Float[ndarray, 'c h w'],
            Float[ndarray, 'c h w'],
            Dict[str, int]
        ]:
        w = self.patch_size

        i = self.lbl.row[index]
        j = self.lbl.col[index]
        cid = self.lbl.data[index].item()

        x_hsi = self.hsi[:, i:i+w, j:j+w]
        x_dsm = self.dsm[:, i:i+w, j:j+w]
        y = np.eye(self.n_class)[cid-1]
        y = torch.from_numpy(y).float()

        # 额外信息: 当前点的坐标
        extras = {
            "location": [i, j],
            "index": index
        }

        return x_hsi, x_dsm, y, extras

    @property
    def uid(self) -> str:
        '''an uid for logging and batch training'''
        return f"{self.INFO['name']}-patchsize{self.patch_size}-len{len(self)}-{self.subset}"

    @property
    def truth(self) -> coo_array:
        '''alias for depreacted property'''
        warnings.warn("`truth` is deprecated, use `lbl` instead", DeprecationWarning)
        return self.lbl
    
    def benchmarker(self) -> Benchmarker:
        return Benchmarker(self.lbl, n_class=self.n_class, dataset_name=self.INFO['name'])
    
    def lbl2rgb(self, lbl, *args, **kwarg):
        return lbl2rgb(lbl, palette=self.INFO['name'], *args, **kwarg)
    
    def hsi2rgb(self, hsi, *args, **kwargs):
        return hsi2rgb(hsi, wavelength=self.INFO['wavelength'], input_format='CHW', *args, **kwargs)
    
    @property
    def composed_rgb(self) -> Float[ndarray, 'c h w']:
        """
        The composed RGB image from HSI.
        :return: RGB image with shape (h, w, c).
        """
        return self.hsi2rgb(self.HSI)