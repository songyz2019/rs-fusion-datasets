from typing import Tuple
import numpy as np
from scipy.sparse import coo_array
from contextlib import contextmanager

@contextmanager
def fixed_seed_rng(seed):
    rng = np.random.default_rng(seed)
    yield rng

def split_spmatrix(a :coo_array, n_sample_perclass=100, seed=0x0d00) -> Tuple[coo_array, coo_array]:
    """
    Split a sparse matrix into train and test sets.
    The train set contains `n_sample_perclass` samples (if n_sample_perclass is int) or percentage of a class (if n_sample_perclass is float) for each class, and the test set contains the rest.
    """
    with fixed_seed_rng(seed) as rng:
        train = coo_array(([],([],[])),a.shape, dtype='int')
        n_class = a.data.max()
        for cid in range(1,n_class+1):
            N = len(a.data[a.data==cid])
            if isinstance(n_sample_perclass, float) and n_sample_perclass < 1.0:
                n_sample_perclass = int(N * n_sample_perclass)
            indice = rng.choice(N, n_sample_perclass, replace=False)
            row = a.row[a.data==cid][indice]
            col = a.col[a.data==cid][indice]
            val = np.ones(len(row)) * cid
            train += coo_array((val, (row, col)), shape=a.shape, dtype='int')
        test = (a - train)
    return train.tocoo(),test.tocoo()
