from typing import List, Tuple, Union, Optional
from pathlib import Path
import numpy as np
from numpy import ndarray
import scipy.io

from scipy.sparse import coo_array
from jaxtyping import  UInt16, Float32, UInt8, Int8, Int4

from ..util.fileio import zip_download_and_extract
from .common import DataMetaInfo


def fetch_houston2018_ouc(
        url       :Union[str, List[str]]      = 'https://github.com/songyz2019/rs-fusion-datasets-dist/releases/download/v1.0.0/houston2018-ouc.zip', 
        data_home :Optional[Union[Path, str]] = None
    ) -> Tuple[
        UInt16[ndarray, '50 1202 4768'], 
        Float32[ndarray, '1 1202 4768'],
        UInt8[coo_array, '1202 4768'],
        UInt8[coo_array, '1202 4768'],
        UInt8[coo_array, '1202 4768'],
        DataMetaInfo
    ]:
    """Fetch and load the Houston2018 (ouc) dataset. 

    The background is 0, and the labels start from 1. All images are CHW formats.
    
    :param url: The URL to download the dataset. Use a list to specify multiple mirrored URLs.
    :param data_home: The path to store the data files, default is SCIKIT_LEARN_DATA environment variable or '~/scikit_learn_data'
    :return: (hsi, lidar, train_labels, train_labels, info)
    """

    basedir = zip_download_and_extract('houston2018-ouc', url, {
        'houston2018-ouc.zip'          :'bc1e046cc00b82661441888b17327b38946a0da33951eb4ebb7d0a4e0ca3cfa2',
        'Houston2018/houston_gt.mat'   : '93f268e576fcf149fab8cb71b4a79691b420e9a3f369f0cbc73e2029639f6286',
        'Houston2018/houston_hsi.mat'  : '35a4a9a2bad193740274c6388f18284fc586a2ef5bde579368796fbeec2a58ea',
        'Houston2018/houston_index.mat': '2d513db197f105b6b82e7039909878862fbe48750a93184017f22ab8f288d0b3',
        'Houston2018/houston_lidar.mat': 'a55fa2d99684078dd0d8039b2ca1349190d86766426a25e8dff7fbd5df036d9b',
    }, data_home=data_home)


    # 3. 数据加载
    hsi   :UInt16[ndarray, '1202 4768 50']= scipy.io.loadmat(
        basedir / 'Houston2018/houston_hsi.mat',
    )['houston_hsi']

    lidar :Float32[ndarray, '1202 4768']   = scipy.io.loadmat(
        basedir / 'Houston2018/houston_lidar.mat',
    )['houston_lidar']
    lidar = lidar[:,:,np.newaxis]

    lbl_map :UInt8[ndarray, '1202 4768']   = scipy.io.loadmat(
        basedir / 'Houston2018/houston_gt.mat',
    )['houston_gt']
    lbl_subsets = scipy.io.loadmat(basedir / 'Houston2018/houston_index.mat')
    lbl_train_coord :Int8[ndarray, 'M 2']    = lbl_subsets['houston_train']   # TODO: replace M N Q with the actual number
    lbl_test_coord  :Int8[ndarray, 'N 2']  = lbl_subsets['houston_test']
    lbl_all_coord   :Int4[ndarray, 'Q 2'] = lbl_subsets['houston_all']
    lbl_train :UInt8[coo_array, '1202 4768'] = coo_array((
        lbl_map[ lbl_train_coord[:,0],lbl_train_coord[:,1] ],
        lbl_train_coord.transpose(1,0),
    ),dtype=lbl_map.dtype,shape=lbl_map.shape)
    lbl_test :UInt8[coo_array, '1202 4768']  = coo_array((
        lbl_map[ lbl_test_coord[:,0],lbl_test_coord[:,1] ],
        lbl_test_coord.transpose(1,0),
    ),dtype=lbl_map.dtype,shape=lbl_map.shape)
    lbl_all  :UInt8[coo_array, '1202 4768']  = coo_array((
        lbl_map[ lbl_all_coord[:,0],lbl_all_coord[:,1] ],
        lbl_all_coord.transpose(1,0),
    ),dtype=lbl_map.dtype,shape=lbl_map.shape)
    info :DataMetaInfo = {
        'name': 'houston2018-ouc',
        'full_name': 'preprocessed houston2018 dataset',
        'version': '',
        'homepage': 'https://github.com/oucailab/DCMNet/',
        'license': '',
        'n_channel_hsi': 50,
        'n_channel_dsm': 1,
        'n_class': 20,
        'width': 4768,
        'height': 1202,
        'label_name': {
            1: 'Healthy grass',
            2: 'Stressed grass',
            3: 'Artificial turf',
            4: 'Evergreen trees',
            5: 'Deciduous trees',
            6: 'Bare earth',
            7: 'Water',
            8: 'Residential buildings',
            9: 'Non-residential buildings',
            10: 'Roads',
            11: 'Sidewalks',
            12: 'Crosswalks',
            13: 'Major thoroughfares',
            14: 'Highways',
            15: 'Railways',
            16: 'Paved parking lots',
            17: 'Unpaved parking lots',
            18: 'Cars',
            19: 'Trains',
            20: 'Stadium seats',
        },
        "wavelength": np.array([
            374.399994,  388.700012,  403.100006,  417.399994,  431.700012,  446.100006,
            460.399994,  474.700012,  489.000000,  503.399994,  517.700012,  532.000000,
            546.299988,  560.599976,  574.900024,  589.200012,  603.599976,  617.900024,
            632.200012,  646.500000,  660.799988,  675.099976,  689.400024,  703.700012,
            718.000000,  732.299988,  746.599976,  760.900024,  775.200012,  789.500000,
            803.799988,  818.099976,  832.400024,  846.700012,  861.099976,  875.400024,
            889.700012,  904.000000,  918.299988,  932.599976,  946.900024,  961.200012,
            975.500000,  989.799988, 1004.200012, 1018.500000, 1032.800049, 1047.099976,
            49.000000,   50.000000
        ])
    }

    return hsi.transpose(2,0,1), lidar.transpose(2,0,1), lbl_train, lbl_test, lbl_all, info



__all__ = ['fetch_trento']
