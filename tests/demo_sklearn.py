# Scikit-Learn focus on the pipeline of data processing, so providing a opionined dataset is not a good idea, 
# Instead, we provide a set of utility functions to help user to create their own dataset
# You should notice that the defualt image format of sklearn is HWC, but the default image format of rs_fusion_datasets is always CHW
# It's a little bit confusing, but we have to do this because we want to keep the same format with PyTorch

import numpy as np
import skimage.io
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_array
from sklearn.preprocessing import OneHotEncoder

from rs_fusion_datasets import fetch_houston2013, patchify, lbl2rgb


HSI, DSM, LBL_TRAIN, LBL_VAL, INFO = fetch_houston2013()
PATCH_SIZE = 5
C_H, H, W  = HSI.shape
C_L        = DSM.shape[0] 
N_TRAIN    = len(LBL_TRAIN.data)
N_VAL      = len(LBL_VAL.data)
N_CLASS    = INFO['n_class']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=32)),
    ('knn', KNeighborsClassifier(n_neighbors=7))
])
onehot = OneHotEncoder(sparse_output=False)

def test():
    pseudo_lbl = coo_array(np.ones(LBL_VAL.shape, LBL_VAL.dtype), dtype=LBL_VAL.dtype)
    (hsi,dsm), _  = patchify([HSI, DSM], pseudo_lbl, PATCH_SIZE)
    x = np.concat([hsi, dsm], axis=1).reshape(hsi.shape[0], -1).astype(np.float32)
    y_pred = pipeline.predict(x)
    y_pred = y_pred.argmax(-1).reshape(H, W)
    skimage.io.imsave('result_sk_y_hat_full.png', lbl2rgb(y_pred+1).transpose(1, 2, 0))


def train():
    (hsi, dsm), y = patchify([HSI, DSM], LBL_TRAIN, PATCH_SIZE) # hsi: (n c h w), dsm: (n c h w), y: (n)
    x = np.concat([hsi, dsm], axis=1).reshape(N_TRAIN, -1).astype(np.float32)   # n c h w -> n (c h w)
    y = onehot.fit_transform(y.reshape(-1, 1))
    pipeline.fit(x, y)


def val():
    (hsi, dsm), y = patchify([HSI, DSM], LBL_VAL, PATCH_SIZE) # hsi: (n c h w), dsm: (n c h w), y: (n d)
    x = np.concat([hsi, dsm], axis=1).reshape(N_VAL, -1).astype(np.float32)  # n c h w -> n (c h w)
    y = onehot.fit_transform(y.reshape(-1, 1))
    y_pred = pipeline.predict(x) # y_pred: (n d)
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train()
    val()
    test()

