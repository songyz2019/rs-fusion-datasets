from pathlib import Path
from typing import Optional, Union
from ..core.fetch_houston2013 import fetch_houston2013
from ..core.fetch_houston2013_mmrs import fetch_houston2013mmrs
from ..core.fetch_muufl import fetch_muufl
from ..core.fetch_trento import fetch_trento
from ..util.split_spmatrix import split_spmatrix
from .common_hsi_dsm_dataset import CommonHsiDsmDataset

class Houston2013(CommonHsiDsmDataset):
    def __init__(self, subset, patch_size=5, data_home :Optional[Union[Path,str]]=None, *args, **kwargs):
        hsi, dsm, lbl_train, lbl_test, info = fetch_houston2013(data_home=data_home)
        super().__init__(hsi, dsm, lbl_train, lbl_test, info, subset, patch_size, *args, **kwargs)

class _Houston2013Mmrs(CommonHsiDsmDataset):
    """This is only for internal test."""
    def __init__(self, subset, patch_size=5, data_home :Optional[Union[Path,str]]=None, *args, **kwargs):
        hsi, dsm, lbl_train, lbl_test, info = fetch_houston2013mmrs(data_home=data_home)
        super().__init__(hsi, dsm, lbl_train, lbl_test, info, subset, patch_size, *args, **kwargs)



class Muufl(CommonHsiDsmDataset):
    """
    
    This dataset is an opinionated version of the MUUFL Gulfport dataset. If you want to use the original dataset, please use `fetch_muufl`. Which tries to keep the original infomation.
    """
    def __init__(self, subset, patch_size=5, data_home :Optional[Union[Path,str]]=None, n_train_perclass:Union[int, float]=100, *args, **kwargs):
        hsi, dsm, lbl, info = fetch_muufl(data_home=data_home)
        lbl_train, lbl_test = split_spmatrix(lbl, n_train_perclass)
        dsm = dsm[0:1]
        super().__init__(hsi, dsm, lbl_train, lbl_test, info, subset, patch_size, *args, **kwargs)

class Trento(CommonHsiDsmDataset):
    def __init__(self, subset, patch_size=5, data_home :Optional[Union[Path,str]]=None, n_train_perclass:Union[int, float]=100, *args, **kwargs):
        hsi, dsm, lbl, info = fetch_muufl(data_home=data_home)
        lbl_train, lbl_test = split_spmatrix(lbl, n_train_perclass)
        dsm = dsm[0:1]
        super().__init__(hsi, dsm, lbl_train, lbl_test, info, subset, patch_size, *args, **kwargs)




__all__ = ['Houston2013', 'Muufl', 'Trento', '_Houston2013Mmrs']