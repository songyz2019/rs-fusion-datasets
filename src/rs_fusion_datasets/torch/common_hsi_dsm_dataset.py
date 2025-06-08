from typing import Literal,Tuple, Dict, Union, Callable
import warnings

from scipy.sparse import coo_array, spmatrix
from torchvision.transforms.v2.functional import crop
import torch
import numpy as np
from numpy import ndarray
from jaxtyping import Num, Float


from ..util.benchmarker import Benchmarker
from ..util.lbl2rgb import lbl2rgb
from ..util.hsi2rgb import hsi2rgb
from ..core.common import DataMetaInfo
from ..util.transforms import Identify, NormalizePerChannel


class CommonHsiDsmDataset(torch.utils.data.Dataset):
    def __init__(self,
                 hsi        :Num[ndarray, 'c h w'], 
                 dsm        :Num[ndarray, 'c h w'],
                 lbl_train  :Num[spmatrix, 'h w'],
                 lbl_test   :Num[spmatrix, 'h w'],
                 info       :DataMetaInfo,
                 split      :Literal['train', 'test', 'full'], 
                 patch_size :int = 5,
                 image_level_preprocess_hsi: Callable[ [Float[ndarray, 'C H W']], Float[ndarray, 'C H W']] = NormalizePerChannel(), # the default NormalizePerChannel will lose information of the original data, but it's the common practice.
                 image_level_preprocess_dsm: Callable[ [Float[ndarray, 'C H W']], Float[ndarray, 'C H W']] = NormalizePerChannel(),
                 image_level_preprocess_lbl: Callable[ [Float[spmatrix, 'H W']],  Float[spmatrix, 'H W']]  = Identify(),
                 n_class: Union[int, Literal['auto','fixed']] = 'fixed',
                 *args, **kwargs):
        """
        
        @param split: 'train', 'test', 'full'. 'full' is not 'train'+'val', it means the whole map, including unclassified labels, with all lbl is -1, usually used for visualization.
        @param image_level_preprocess_hsi: image level preprocess for hsi
        @param image_level_preprocess_dsm: image level preprocess for dsm
        @param image_level_preprocess_lbl: image level preprocess for label, usually a mapping function to convert labels to a specific format
        @param n_class: If you do not use image_level_preprocess_lbl, leave it default.Number of classes in the dataset. If 'auto', it will be determined from the labels. If 'fixed', it will use the number of classes from the info dictionary. If an int, it will be used as the number of classes. 
        """
        super().__init__(*args, **kwargs)
        
        # Load truth
        self.split = split
        if self.split == 'train':
            self.lbl = lbl_train
        elif self.split == 'test':
            self.lbl = lbl_test
        elif self.split == 'full':
            self.lbl = coo_array(-1*np.ones(lbl_test.shape, dtype=np.int16), dtype=np.int16)
        else:
            raise ValueError(f"Unknown dataset split: {split}")

        # Load patch size
        self.patch_size   = patch_size
        self.patch_radius = patch_size // 2 # if patch_size is odd, it should be (patch_size - w_center)/2, but some user will use on-odd patch size


        # Load basic info
        self.HSI, self.DSM, self.INFO = hsi, dsm, info

        # Preprocess HSI
        self.image_level_preprocess_hsi = image_level_preprocess_hsi
        pad_shape = ((0, 0), (self.patch_radius, self.patch_radius), (self.patch_radius, self.patch_radius))
        self.hsi = self.image_level_preprocess_hsi(self.HSI)
        self.hsi = np.pad(self.hsi, pad_shape, mode='reflect')
        self.hsi = torch.from_numpy(self.hsi).float()
        
        # Preprocess DSM
        self.image_level_preprocess_dsm = image_level_preprocess_dsm
        self.dsm = self.image_level_preprocess_dsm(self.DSM)
        self.dsm = np.pad(self.dsm, pad_shape, mode='reflect')
        self.dsm = torch.from_numpy(self.dsm).float()

        # Preprocess label
        self.image_level_preprocess_lbl = image_level_preprocess_lbl
        self.lbl = self.image_level_preprocess_lbl(self.lbl)

        # util members for users
        if n_class == 'auto':
            self.n_class = self.lbl.data.max() + 1
        elif n_class == 'fixed':
            self.n_class = self.INFO['n_class']
        elif isinstance(n_class, int):
            assert n_class > 0
            self.n_class = n_class
        else:
            raise ValueError(f"Unknown n_class: {n_class}, it must be 'auto', 'fixed' or an int")
        self.n_channel_hsi = self.hsi.shape[0]
        self.n_channel_dsm = self.dsm.shape[0]

        # cache for one-hot encoding in __getitem__
        self.onehot_eye = torch.eye(self.n_class).float()

    def __len__(self):
        return self.lbl.count_nonzero()

    def __getitem__(self, index) -> Tuple[
            Float[ndarray, 'c h w'],
            Float[ndarray, 'c h w'],
            Float[ndarray, 'c h w'],
            Dict[str, int]
        ]:
        """
        Get a patch of HSI and DSM, and the corresponding label.
        The lbl is a pesudo one-hot encoding value when split is 'full', you should only use it to avoid errors in your model if your loss and in your model.
        """
        w = self.patch_size

        i = self.lbl.row[index]
        j = self.lbl.col[index]
        cid = self.lbl.data[index].item()

        # x_hsi = self.hsi[:, i:i+w, j:j+w]
        # x_dsm = self.dsm[:, i:i+w, j:j+w]
        # y = np.eye(self.n_class)[cid-1]
        # y = torch.from_numpy(y).float()
        x_hsi = crop(self.hsi, i, j, w, w) # since the image is padded, the i j is just the top-left corner of the patch
        x_dsm = crop(self.dsm, i, j, w, w)
        y = self.onehot_eye[cid-1] # cid is 1-based, so we need to -1

        extras = {
            "location": [i, j],
            "index": index
        }

        return x_hsi, x_dsm, y, extras

    @property
    def uid(self) -> str:
        '''an uid for logging and batch training'''
        return f"{self.INFO['name']}-patchsize{self.patch_size}-len{len(self)}-{self.split}"

    @property
    def truth(self) -> coo_array:
        '''alias for depreacted property'''
        warnings.warn("`truth` is deprecated, use `lbl` instead", DeprecationWarning)
        return self.lbl
    
    def benchmarker(self) -> Benchmarker:
        return Benchmarker(self.lbl, n_class=self.n_class, dataset_name=self.INFO['name'])
    
    def lbl2rgb(self, lbl=None, *args, **kwarg):
        if lbl is None:
            lbl = self.lbl
        return lbl2rgb(lbl, palette=self.INFO['name'], *args, **kwarg)
    
    def hsi2rgb(self, hsi=None, to_u8np=True,*args, **kwargs):
        if hsi is None:
            hsi = self.HSI
        return hsi2rgb(hsi, wavelength=self.INFO['wavelength'], input_format='CHW', output_format='CHW', to_u8np=to_u8np, *args, **kwargs)
    