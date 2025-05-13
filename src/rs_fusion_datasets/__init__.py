import logging

from .core.fetch_houston2013 import fetch_houston2013
from .core.fetch_houston2013_mmr import fetch_houston2013_mmr
from .core._fetch_houston2013_mat import _fetch_houston2013_mat
from .core.fetch_muufl import fetch_muufl
from .core.fetch_trento import fetch_trento
from .core.fetch_augsburg_ouc import fetch_augsburg_ouc
from .core.fetch_berlin_ouc import fetch_berlin_ouc
from .core.fetch_houston2018_ouc import fetch_houston2018_ouc
from .core._fetch_trento_mat import _fetch_trento_mat
from .core._fetch_muufl_mat import _fetch_muufl_mat
from .core.common import DataMetaInfo

from .util.split_spmatrix import split_spmatrix
from .util.fileio import read_roi
from .util.hsi2rgb import hsi2rgb


__all__ = [
    'fetch_houston2013', 
    'fetch_houston2013_mmr', 
    'fetch_muufl', 
    'fetch_trento', 
    'fetch_berlin_ouc',
    'fetch_augsburg_ouc', 
    'fetch_houston2018_ouc',
    '_fetch_trento_mat',
    '_fetch_muufl_mat',
    '_fetch_houston2013_mat',
    'split_spmatrix', 
    'read_roi', 
    'hsi2rgb',
    'DataMetaInfo',
]


# If torch is imported, add these API
try:
    import torch
except ImportError:
    pass
else:
    from .torch.datasets import Houston2013, Muufl, Trento, Houston2013Mmr, _Houston2013Mmrs, BerlinOuc, AugsburgOuc, Houston2018Ouc, _MuuflMat, _TrentoMat, _Houston2013Mat
    from .torch.common_hsi_dsm_dataset import CommonHsiDsmDataset
    from .util.lbl2rgb import lbl2rgb
    from .util.classification_mapper import ClassificationMapper

    __all__ += [
        'CommonHsiDsmDataset', 
        'Houston2013',
        'Houston2013Mmr', 
        'Muufl', 
        'Trento', 
        'BerlinOuc', 
        'AugsburgOuc',
        'Houston2018Ouc',
        '_MuuflMat', 
        '_TrentoMat', 
        '_Houston2013Mmrs',
        '_Houston2013Mat',
        'lbl2rgb',
        'ClassificationMapper'
    ]





