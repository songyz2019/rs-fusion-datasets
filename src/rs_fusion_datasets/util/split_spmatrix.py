import numpy as np
from scipy.sparse import coo_array

def split_spmatrix(a, n_samples=20, seed=0x0d000721):
    np.random.seed(seed)
    train = coo_array(([],([],[])),a.shape, dtype='int')
    n_class = a.data.max()
    for cid in range(1,n_class+1):
        N = len(a.data[a.data==cid])
        indice = np.random.choice(N, n_samples, replace=False)
        row = a.row[a.data==cid][indice]
        col = a.col[a.data==cid][indice]
        val = np.ones(len(row)) * cid
        train += coo_array((val, (row, col)), shape=a.shape, dtype='int')
    test = (a - train)
    return train.tocoo(),test.tocoo()
