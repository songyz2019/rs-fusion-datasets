from ..core.fetch_houston2013 import fetch_houston2013
from ..core.fetch_muufl import fetch_muufl
from ..core.fetch_trento import fetch_trento
from ..core._fetch_houston2013mmrs import _fetch_houston2013mmrs
from ..util.split_spmatrix import split_spmatrix
from .common_hsi_dsm_dataset import CommonHsiDsmDataset

class Houston2013(CommonHsiDsmDataset):
    def __init__(self, subset, patch_size=5, *args, **kwargs):
        super().__init__(fetch_houston2013, subset, patch_size, *args, **kwargs)

class _Houston2013Mmrs(CommonHsiDsmDataset):
    """This is only for internal test."""
    def __init__(self, subset, patch_size=5, *args, **kwargs):
        super().__init__(_fetch_houston2013mmrs, subset, patch_size, *args, **kwargs)


class Muufl(CommonHsiDsmDataset):
    """
    
    This dataset is an opinionated version of the MUUFL Gulfport dataset. If you want to use the original dataset, please use `fetch_muufl`. Which tries to keep the original infomation.
    """
    def __init__(self, subset, patch_size=5, n_train_perclass=100, *args, **kwargs):
        def get_data():
            hsi, dsm, truth, info = fetch_muufl()
            train_truth, test_truth = split_spmatrix(truth, n_train_perclass)
            return hsi, dsm[0:1], train_truth, test_truth, info
        super().__init__(get_data, subset, patch_size, *args, **kwargs)

class Trento(CommonHsiDsmDataset):
    def __init__(self, subset, patch_size=5, n_train_perclass=100, *args, **kwargs):
        def get_data():
            hsi, dsm, truth, info = fetch_trento()
            train_truth, test_truth = split_spmatrix(truth, n_train_perclass)
            return hsi, dsm[0:1], train_truth, test_truth, info
        super().__init__(get_data, subset, patch_size, *args, **kwargs)



__all__ = ['Houston2013', 'Muufl', 'Trento', '_Houston2013Mmrs']