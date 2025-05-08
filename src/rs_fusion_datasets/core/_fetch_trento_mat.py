from typing import List, Tuple, Union, Optional
from pathlib import Path
import numpy as np
from numpy import ndarray
import scipy.io

from scipy.sparse import coo_array

from ..util.fileio import zip_download_and_extract, load_one_key_mat
from .common import DataMetaInfo
from jaxtyping import  Float32, UInt8


def _fetch_trento_mat(
        url       :Union[str, List[str]]      = 'http://10.7.36.2:5000/dataset/trento_mat.zip', 
        data_home :Optional[Union[Path, str]] = None
        ) -> Tuple[
        Float32[ndarray, '63 166 600'], 
        Float32[ndarray, '2 166 600'],
        UInt8[coo_array, '166 600'],
        DataMetaInfo
    ]:
    """Fetch and load the Trento dataset.

    The background is 0, and the labels start from 1. All images are CHW formats.
    
    :param url: The URL to download the dataset. Use a list to specify multiple mirrored URLs.
    :param data_home: The path to store the data files, default is SCIKIT_LEARN_DATA environment variable or '~/scikit_learn_data'
    :return: (hsi, lidar, train_labels, train_labels, info)
    """

    basedir = zip_download_and_extract('trento-mat', url, {
        'trento-mat.zip'        : 'ae8a2e498a51ca0aad8995e86f1533390c880197a8a9cd96b7d28fe6e962274b',
        'Trento/HSI.mat'    : '44a84b4fe8bab9b2f5ec11af32602436ccc97653b7ae8c3283334da7d221c053',
        'Trento/LiDAR.mat'  : 'd99347532e7b18982c9537fd4de43574dd64f9b35ccab251459e7b483ddd43df',
        'Trento/TRLabel.mat': '593976a1bef4edf06ec543926811509f997454557aa9313ed6165b3012c26e07',
        'Trento/TSLabel.mat': '629ae33464907098cacf58549996711bf7079f036946f9208fde8a67842ab499'
    }, data_home=data_home)


    # 3. 数据加载
    hsi  :Float32[ndarray, '166 600 63']  = load_one_key_mat(
        basedir / 'Trento/HSI.mat',
    )

    lidar :Float32[ndarray, '166 600']  = load_one_key_mat(
        basedir / 'Trento/LiDAR.mat',
    )
    lidar = np.expand_dims(lidar, axis=-1)

    lbl_train :UInt8[ndarray, '166 600'] = load_one_key_mat(
        basedir / 'Trento/TRLabel.mat',
    )
    lbl_train = coo_array(lbl_train)

    lbl_test :UInt8[ndarray, '166 600'] = load_one_key_mat(
        basedir / 'Trento/TSLabel.mat',
    )
    lbl_test = coo_array(lbl_test)


    info :DataMetaInfo = {
        'name': 'trento-mat',
        'full_name': 'Trento',
        'version': '0.1',
        'homepage': '',
        'license': '',
        'n_channel_hsi': hsi.shape[-1],
        'n_channel_dsm': lidar.shape[-1],
        'n_class': 6,
        'width': hsi.shape[1],
        'height': hsi.shape[0],
        'label_name': {
            1: "Apple Trees",
            2: "Building",
            3: "Ground",
            4: "Woods",
            5: "Vineyard",
            6: "Roads"
        },
        'wavelength': np.linspace(402.89, 989.09, hsi.shape[-1])
    }

    return hsi.transpose(2,0,1), lidar.transpose(2,0,1), lbl_train, lbl_test, info



__all__ = ['fetch_trento']
