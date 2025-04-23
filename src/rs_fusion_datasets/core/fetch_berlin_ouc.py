from typing import List, Tuple, Union, Optional
from pathlib import Path
import numpy as np
from numpy import ndarray
import scipy.io

from scipy.sparse import coo_array
from jaxtyping import  UInt16, Float64, UInt8, Int8, Int4

from ..util.fileio import zip_download_and_extract
from .common import DataMetaInfo


def fetch_berlin_ouc(
        url       :Union[str, List[str]]      = 'https://github.com/songyz2019/rs-fusion-datasets-dist/releases/download/v1.0.0/berlin-ouc.zip', 
        data_home :Optional[Union[Path, str]] = None
    ) -> Tuple[
        UInt16[ndarray, '180 1723 476'], 
        Float64[ndarray, '4 1723 476'],
        UInt8[coo_array, '1723 476'],
        UInt8[coo_array, '1723 476'],
        UInt8[coo_array, '1723 476'],
        DataMetaInfo
    ]:
    """Fetch and load the Berlin (ouc) dataset. 

    The background is 0, and the labels start from 1. All images are CHW formats.
    
    :param url: The URL to download the dataset. Use a list to specify multiple mirrored URLs.
    :param data_home: The path to store the data files, default is SCIKIT_LEARN_DATA environment variable or '~/scikit_learn_data'
    :return: (hsi, lidar, train_labels, train_labels, info)
    """

    basedir = zip_download_and_extract('berlin-ouc', url, {
        'berlin-ouc.zip'         : '5fdd156b0283bddab3cf9916f34d3d6b3a2ceb68e7c38c3f8022292199aafcf8',
        'Berlin/berlin_gt.mat'   : 'dd9286cd565535857721627878de3cbebffa2dacc625b539474d16cba3a7343e',
        'Berlin/berlin_hsi.mat'  : '580f397d9f249b4ec3f02b8cc570d910d765afa9f77b17d8a4f4799edee94b1f',
        'Berlin/berlin_index.mat': '1a66ec454ea95bbf5782525d9fcf9a04412a948b7b3aa6919ed324516d42a8b5',
        'Berlin/berlin_sar.mat'  : 'a6cf2f86db7f1a02c340e813a7b46f9c6c58e297e1eb9b5fa19ed49244b2b8a7'
    }, data_home=data_home)


    # 3. 数据加载
    hsi   :Float64[ndarray, '1723 476 244']= scipy.io.loadmat(
        basedir / 'Berlin/berlin_hsi.mat',
    )['berlin_hsi']

    sar :Float64[ndarray, '1723 476 4']   = scipy.io.loadmat(
        basedir / 'Berlin/berlin_sar.mat',
    )['berlin_sar']

    lbl_map :UInt8[ndarray, '1723 476']   = scipy.io.loadmat(
        basedir / 'Berlin/berlin_gt.mat',
    )['berlin_gt']
    lbl_subsets = scipy.io.loadmat(basedir / 'Berlin/berlin_index.mat')
    lbl_train_coord :Int8[ndarray, 'M 2']    = lbl_subsets['berlin_train']   # TODO: replace M N Q with the actual number
    lbl_test_coord  :Int8[ndarray, 'N 2']  = lbl_subsets['berlin_test']
    lbl_all_coord   :Int4[ndarray, 'Q 2'] = lbl_subsets['berlin_all']
    lbl_train :UInt8[coo_array, '1723 476'] = coo_array((
        lbl_map[ lbl_train_coord[:,0],lbl_train_coord[:,1] ],
        lbl_train_coord.transpose(1,0),
    ),dtype=lbl_map.dtype,shape=lbl_map.shape)
    lbl_test :UInt8[coo_array, '1723 476']  = coo_array((
        lbl_map[ lbl_test_coord[:,0],lbl_test_coord[:,1] ],
        lbl_test_coord.transpose(1,0),
    ),dtype=lbl_map.dtype,shape=lbl_map.shape)
    lbl_all  :UInt8[coo_array, '1723 476']  = coo_array((
        lbl_map[ lbl_all_coord[:,0],lbl_all_coord[:,1] ],
        lbl_all_coord.transpose(1,0),
    ),dtype=lbl_map.dtype,shape=lbl_map.shape)

    info :DataMetaInfo = {
        'name': 'berlin-ouc',
        'full_name': 'preprocessed berlin dataset',
        'version': '',
        'homepage': 'https://github.com/oucailab/DCMNet/',
        'license': '',
        'n_channel_hsi': 244,
        'n_channel_dsm': 4,
        'n_class': 9,
        'width': 1723,
        'height': 476,
        'label_name': {
            1: 'Forest',
            2: 'Residential',
            3: 'Industrial',
            4: 'Lowplants',
            5: 'Soil',
            6: 'Allotment',
            7: 'Commercial',
            8: 'Water'
        },
        'wavelength': np.linspace(400, 2500, 244) # TODO: check the wavelength
    }

    return hsi.transpose(2,0,1), sar.transpose(2,0,1), lbl_train, lbl_test, lbl_all, info



__all__ = ['fetch_trento']
