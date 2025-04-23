from pathlib import Path
from typing import Optional, Union, Literal
import warnings
from ..core.fetch_houston2013 import fetch_houston2013
from ..core.fetch_houston2013_mmrs import fetch_houston2013_mmr
from ..core.fetch_muufl import fetch_muufl
from ..core.fetch_trento import fetch_trento
from ..core.fetch_augsburg_ouc import fetch_augsburg_ouc
from ..core.fetch_berlin_ouc import fetch_berlin_ouc
from ..core.fetch_houston2018_ouc import fetch_houston2018_ouc
from ..util.split_spmatrix import split_spmatrix
from .common_hsi_dsm_dataset import CommonHsiDsmDataset

class Houston2018Ouc(CommonHsiDsmDataset):
    def __init__(self, split: Literal['train', 'test', 'full'], patch_size=5, root :Optional[Union[Path,str]]=None, *args, **kwargs):
        """
        A preprocessed torch dataset for Houston 2018 (ouc) dataset.

        :param split: 'train', 'test', 'full'. 'full' means the whole map, usually used for visualization.
        :param patch_size: The size of patches. Default is 5.
        :param root: The path to store the data files, default is SCIKIT_LEARN_DATA environment variable or '~/scikit_learn_data'
        """
        hsi, dsm, lbl_train, lbl_test, lbl_all,info = fetch_houston2018_ouc(data_home=root)
        super().__init__(hsi, dsm, lbl_train, lbl_test, info, split, patch_size, *args, **kwargs)


class BerlinOuc(CommonHsiDsmDataset):
    """
    A preprocessed torch dataset for Berlin (ouc) dataset.

    :param split: 'train', 'test', 'full'. 'full' means the whole map, usually used for visualization.
    :param patch_size: The size of patches. Default is 5.
    :param root: The path to store the data files, default is SCIKIT_LEARN_DATA environment variable or '~/scikit_learn_data'
    """
    def __init__(self, split: Literal['train', 'test', 'full'], patch_size=5, root :Optional[Union[Path,str]]=None, *args, **kwargs):
        hsi, dsm, lbl_train, lbl_test, lbl_all,info = fetch_berlin_ouc(data_home=root)
        super().__init__(hsi, dsm, lbl_train, lbl_test, info, split, patch_size, *args, **kwargs)

class AugsburgOuc(CommonHsiDsmDataset):
    """
    A preprocessed torch dataset for the Augsburg 2018 (ouc) dataset.

    :param split: 'train', 'test', 'full'. 'full' means the whole map, usually used for visualization.
    :param patch_size: The size of patches. Default is 5.
    :param root: The path to store the data files, default is SCIKIT_LEARN_DATA environment variable or '~/scikit_learn_data'
    """
    def __init__(self, split: Literal['train', 'test', 'full'], patch_size=5, root :Optional[Union[Path,str]]=None, *args, **kwargs):
        hsi, dsm, lbl_train, lbl_test, lbl_all,info = fetch_augsburg_ouc(data_home=root)
        super().__init__(hsi, dsm, lbl_train, lbl_test, info, split, patch_size, *args, **kwargs)

class Houston2013(CommonHsiDsmDataset):
    """
    A preprocessed torch dataset for the official Houston 2013 dataset.

    :param split: 'train', 'test', 'full'. 'full' means the whole map, usually used for visualization.
    :param patch_size: The size of patches. Default is 5.
    :param root: The path to store the data files, default is SCIKIT_LEARN_DATA environment variable or '~/scikit_learn_data'
    """
    def __init__(self, split: Literal['train', 'test', 'full'], patch_size=5, root :Optional[Union[Path,str]]=None, *args, **kwargs):
        hsi, dsm, lbl_train, lbl_test, info = fetch_houston2013(data_home=root)
        super().__init__(hsi, dsm, lbl_train, lbl_test, info, split, patch_size, *args, **kwargs)

class _Houston2013Mmrs(CommonHsiDsmDataset):
    """This is only for internal test."""
    def __init__(self, split: Literal['train', 'test', 'full'], patch_size=5, root :Optional[Union[Path,str]]=None, *args, **kwargs):
        hsi, dsm, lbl_train, lbl_test, info = fetch_houston2013_mmr(data_home=root)
        super().__init__(hsi, dsm, lbl_train, lbl_test, info, split, patch_size, *args, **kwargs)



class Muufl(CommonHsiDsmDataset):
    """
    A preprocessed torch dataset for the official Muufl dataset.

    :param split: 'train', 'test', 'full'. 'full' means the whole map, usually used for visualization.
    :param patch_size: The size of patches. Default is 5.
    :param n_train_perclass: The number of training samples per class. Default is 100.
    :param root: The path to store the data files, default is SCIKIT_LEARN_DATA environment variable or '~/scikit_learn_data'
    """
    def __init__(self, split: Literal['train', 'test', 'full'], patch_size=5, root :Optional[Union[Path,str]]=None, n_train_perclass:Union[int, float]=100, *args, **kwargs):
        hsi, dsm, lbl, info = fetch_muufl(data_home=root)
        lbl_train, lbl_test = split_spmatrix(lbl, n_train_perclass)
        super().__init__(hsi, dsm, lbl_train, lbl_test, info, split, patch_size, *args, **kwargs)

class Trento(CommonHsiDsmDataset):
    """
    A preprocessed torch dataset for the Trento dataset.

    :param split: 'train', 'test', 'full'. 'full' means the whole map, usually used for visualization.
    :param patch_size: The size of patches. Default is 5.
    :param n_train_perclass: The number of training samples per class. Default is 100.
    :param root: The path to store the data files, default is SCIKIT_LEARN_DATA environment variable or '~/scikit_learn_data'
    """
    def __init__(self, split: Literal['train', 'test', 'full'], patch_size=5, root :Optional[Union[Path,str]]=None, n_train_perclass:Union[int, float]=100, *args, **kwargs):
        hsi, dsm, lbl, info = fetch_trento(data_home=root)
        lbl_train, lbl_test = split_spmatrix(lbl, n_train_perclass)
        super().__init__(hsi, dsm, lbl_train, lbl_test, info, split, patch_size, *args, **kwargs)




__all__ = ['Houston2013', 'Muufl', 'Trento', '_Houston2013Mmrs']