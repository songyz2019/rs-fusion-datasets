import logging

from .core.fetch_houston2013 import fetch_houston2013
from .core._fetch_houston2018 import fetch_houston2018
from .core.fetch_houston2013_mmrs import fetch_houston2013mmrs
from .core.fetch_muufl import fetch_muufl
from .core.fetch_trento import fetch_trento
from .core.common import DataMetaInfo

from .util.split_spmatrix import split_spmatrix
from .util.fileio import read_roi


__all__ = ['fetch_houston2013', 'fetch_houston2013mmrs', 'fetch_houston2018','fetch_muufl', 'fetch_trento', 'split_spmatrix', 'read_roi', 'DataMetaInfo']


# If torch is imported, add these API
try:
    import torch
except ImportError:
    pass
else:
    from .torch.datasets import Houston2013, Muufl, Trento, _Houston2013Mmrs
    from .torch.common_hsi_dsm_dataset import CommonHsiDsmDataset
    from .util.lbl2rgb import lbl2rgb

    __all__ += ['CommonHsiDsmDataset', 'Houston2013','_Houston2013Mmrs', 'Muufl', 'Trento', 'lbl2rgb']





