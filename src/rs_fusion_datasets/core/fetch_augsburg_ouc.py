from typing import List, Tuple, Union, Optional
from pathlib import Path
import numpy as np
from numpy import ndarray
import scipy.io

from scipy.sparse import coo_array
from jaxtyping import  UInt16, Float64, UInt8, Int8, Int4

from ..util.fileio import zip_download_and_extract
from .common import DataMetaInfo


def fetch_augsburg_ouc(
        url       :Union[str, List[str]]      = 'https://github.com/songyz2019/rs-fusion-datasets-dist/releases/download/v1.0.0/augsburg-ouc.zip', 
        data_home :Optional[Union[Path, str]] = None
    ) -> Tuple[
        UInt16[ndarray, '180 332 485'], 
        Float64[ndarray, '4 332 485'],
        UInt8[coo_array, '332 485'],
        UInt8[coo_array, '332 485'],
        UInt8[coo_array, '332 485'],
        DataMetaInfo
    ]:
    """Fetch and load the Augsburg (ouc) dataset. 

    The background is 0, and the labels start from 1. All images are CHW formats.
    
    :param url: The URL to download the dataset. Use a list to specify multiple mirrored URLs.
    :param data_home: The path to store the data files, default is SCIKIT_LEARN_DATA environment variable or '~/scikit_learn_data'
    :return: (hsi, lidar, train_labels, train_labels, info)
    """

    basedir = zip_download_and_extract('augsburg-ouc', url, {
        'augsburg-ouc.zip'            :'4d853be609eca98a5fb23ccd711bcd0985b7db86a6e89f53f461e1643a7ab6ae',
        'Augsburg/augsburg_gt.mat'    : '065df71257f06c075ff3b37327c0fba75cc9146015a0f095a093fd12b9eaacbe',
        'Augsburg/augsburg_hsi.mat'   : '2f13839d7840b32f55e0489d6934776dcfded7f678e23a9d8c2c092573946583',
        'Augsburg/augsburg_index.mat' : '8bc57037447cb42dbf3a8478d42d4bc217a53a6a0b349c2d67db8c660bef2f43',
        'Augsburg/augsburg_sar.mat'   : 'cd1dbfad2346f47e51d105b05e11aaefc872fe487a94bdc2913f4eaee3f8e82b'
    }, data_home=data_home)


    # 3. 数据加载
    hsi   :UInt16[ndarray, '332 485 180']= scipy.io.loadmat(
        basedir / 'Augsburg/augsburg_hsi.mat',
    )['augsburg_hsi']

    sar :Float64[ndarray, '332 485 4']   = scipy.io.loadmat(
        basedir / 'Augsburg/augsburg_sar.mat',
    )['augsburg_sar']

    lbl_map :UInt8[ndarray, '332 485']   = scipy.io.loadmat(
        basedir / 'Augsburg/augsburg_gt.mat',
    )['augsburg_gt']
    lbl_subsets = scipy.io.loadmat(basedir / 'Augsburg/augsburg_index.mat')
    lbl_train_coord :Int8[ndarray, '761 2']    = lbl_subsets['augsburg_train'] 
    lbl_test_coord  :Int8[ndarray, '77533 2']  = lbl_subsets['augsburg_test']
    lbl_all_coord   :Int4[ndarray, '161020 2'] = lbl_subsets['augsburg_all']
    lbl_train :UInt8[coo_array, '332 485'] = coo_array((
        lbl_map[ lbl_train_coord[:,0],lbl_train_coord[:,1] ],
        lbl_train_coord.transpose(1,0),
    ),dtype=lbl_map.dtype,shape=lbl_map.shape)
    lbl_test :UInt8[coo_array, '332 485']  = coo_array((
        lbl_map[ lbl_test_coord[:,0],lbl_test_coord[:,1] ],
        lbl_test_coord.transpose(1,0),
    ),dtype=lbl_map.dtype,shape=lbl_map.shape)
    lbl_all  :UInt8[coo_array, '332 485']  = coo_array((
        lbl_map[ lbl_all_coord[:,0],lbl_all_coord[:,1] ],
        lbl_all_coord.transpose(1,0),
    ),dtype=lbl_map.dtype,shape=lbl_map.shape)

    info :DataMetaInfo = {
        'name': 'augsburg-ouc',
        'full_name': 'preprocessed augsburg dataset',
        'version': '',
        'homepage': 'https://github.com/oucailab/DCMNet/',
        'license': '',
        'n_channel_hsi': 180,
        'n_channel_dsm': 4,
        'n_class': 8,
        'width': 485,
        'height': 332,
        'label_name': {
            1: 'Forest',
            2: 'Residential',
            3: 'Industrial',
            4: 'Lowplants',
            5: 'Soil',
            6: 'Allotment',
            7: 'Commercial',
            8: 'Water',

        },
        'wavelength': np.linspace(410, 2500, 180) # TODO: check the wavelength
    }

    return hsi.transpose(2,0,1), sar.transpose(2,0,1), lbl_train, lbl_test, lbl_all, info



__all__ = ['fetch_trento']
