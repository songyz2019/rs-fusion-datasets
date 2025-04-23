from typing import List, Tuple, Union, Optional
from pathlib import Path
import numpy as np
from numpy import ndarray
import scipy.io

from scipy.sparse import coo_array

from ..util.fileio import zip_download_and_extract
from .common import DataMetaInfo
from jaxtyping import  Float32, UInt8


def fetch_trento(
        url       :Union[str, List[str]]      = 'https://github.com/tyust-dayu/Trento/archive/b4afc449ce5d6936ddc04fe267d86f9f35536afd.zip', 
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

    basedir = zip_download_and_extract('trento', url, {
        'trento.zip'      :'b203331b039d994015c4137753f15973cb638046532b8dced6064888bf970631',
        'Trento-b4afc449ce5d6936ddc04fe267d86f9f35536afd/allgrd.mat'      :'7e3fb2a2ea22c2661dfc768db3cb93c9643b324e7e64fadedfa57f5edbf1818f',
        'Trento-b4afc449ce5d6936ddc04fe267d86f9f35536afd/Italy_hsi.mat'   :'7b965fd405314b5c91451042e547a1923be6f5a38c6da83969032cff79729280',
        'Trento-b4afc449ce5d6936ddc04fe267d86f9f35536afd/Italy_lidar.mat' :'a04dc90368d6a7b4f9d3936024ba9fef4105456c090daa14fff31b8b79e94ab1',
    }, data_home=data_home)


    # 3. 数据加载
    hsi  :Float32[ndarray, '166 600 63']  = scipy.io.loadmat(
        basedir / 'Trento-b4afc449ce5d6936ddc04fe267d86f9f35536afd/Italy_hsi.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['data']

    lidar :Float32[ndarray, '166 600 2']  = scipy.io.loadmat(
        basedir / 'Trento-b4afc449ce5d6936ddc04fe267d86f9f35536afd/Italy_lidar.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['data']

    truth :UInt8[ndarray, '166 600'] = scipy.io.loadmat(
        basedir / 'Trento-b4afc449ce5d6936ddc04fe267d86f9f35536afd/allgrd.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['mask_test']
    truth = coo_array(truth)

    info :DataMetaInfo = {
        'name': 'trento',
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

    return hsi.transpose(2,0,1), lidar.transpose(2,0,1), truth, info



__all__ = ['fetch_trento']
