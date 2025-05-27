from typing import Tuple, Union
import numpy as np
from scipy.sparse import coo_array
from contextlib import contextmanager

@contextmanager
def fixed_seed_rng(seed):
    rng = np.random.default_rng(seed)
    yield rng

def split_spmatrix(a :coo_array, n_sample_perclass :Union[int, float]=100, seed=0x0d00) -> Tuple[coo_array, coo_array]:
    """
    Split a sparse matrix into train and test sets.
    
    The train set contains `n_sample_perclass` samples (if n_sample_perclass is int) or percentage of a class (if n_sample_perclass is float) for each class, and the test set contains the rest.
    At least one sample in testset is guaranteed when using ratio n_sample_perclass. but not the case for trainset.
    The random seed is fixed to ensure reproducibility, when loading trainset and testset.

    @param a: Sparse matrix in COO format.
    @param n_sample_perclass: Number of samples per class for the train set. If a float, it is treated as a ratio of the total number of samples in each class.
    @return: Tuple of train and test sparse matrices in COO format.
    """
    with fixed_seed_rng(seed) as rng:
        train = coo_array(([],([],[])),a.shape, dtype='int')
        n_class = a.data.max()
        is_ratio = isinstance(n_sample_perclass, float) and n_sample_perclass <= 1.0
        for cid in range(1,n_class+1):
            N = len(a.data[a.data==cid])
            if is_ratio:
                n = int(N * n_sample_perclass)
            else:
                n = int(n_sample_perclass)
            if n == 0:
                continue
            if n > N:
                n = N
            indice = rng.choice(N, n, replace=False)
            row = a.row[a.data==cid][indice]
            col = a.col[a.data==cid][indice]
            val = np.ones(len(row)) * cid
            train += coo_array((val, (row, col)), shape=a.shape, dtype='int')
        test = (a - train)
    return train.tocoo(),test.tocoo()
