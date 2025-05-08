# SPDX-FileCopyrightText: 2025-present songyz2023 <songyz2023dlut@outlook.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union,Optional
from pathlib import Path

import numpy as np
from scipy.sparse import coo_array, spmatrix
from jaxtyping import UInt16, Float32, UInt64, Float64

from ..util.fileio import load_one_key_mat, zip_download_and_extract
from .common import DataMetaInfo

def _fetch_houston2013_mat(
    url         :Union[str, List[str]]      = 'http://10.7.36.2:5000/dataset/houston2013_mat.zip', 
    data_home   :Optional[Union[Path, str]] = None
) -> tuple[
    UInt16[np.ndarray, '144 349 1905'],
    Float32[np.ndarray, '1 349 1905'],
    UInt64[spmatrix, '349 1905'],
    UInt64[spmatrix, '349 1905'],
    DataMetaInfo
]:
    """Fetch and load the Houston2013 dataset.

    The background is 0, and the labels start from 1. All images are CHW formats.
    
    :param url: The URL to download the dataset. Use a list to specify multiple mirrored URLs.
    :param url_lbl_val: The URL to download the validation labels of the dataset. Use a list to specify multiple mirrored URLs.
    :param data_home: The path to store the data files, default is SCIKIT_LEARN_DATA environment variable or '~/scikit_learn_data'
    :return: (hsi, lidar, train_labels, train_labels, info)
    """
    basedir = zip_download_and_extract('houston2013-mat', url, 
        {
            'houston2013-mat.zip': '6ecc03e6fc886bd1cc76a4693200b2bb600af7d735149fa324f1ed63ee0621a9',
            'Houston2013/DSM.mat':   '3e65d6405a4113d5881e30f0fd3c4fee743d90a1e51005dbf4162a493dfa4ff1',
            'Houston2013/HSI.mat':   '8860bbe72c27fdbf0820824c24deb7cfba95231618e98376264c4ab09b6882b3',
            'Houston2013/mask_TE.mat':   'b70c2af8c767ad119b2c9edbba3842cc2840aece621a64c0a073caf7d593770c',
            'Houston2013/mask_TR.mat':   'f957b05b8c0f29205933222338df05a3954e16217e44557d44924eed50ff8111',
            'Houston2013/TE.mat':   'f928d94308928978c5abdd6ef679e78382273f411c87970cc0c7a1f429cdd443',
            'Houston2013/TR.mat':   '93ada06aaf1bd5e5363ab7ef507f59c08e46b4227f48a077fea10fe54465a41d',
        },
        data_home=data_home
    )

        
    # 3. 数据加载
    hsi: Float64[np.ndarray, '349 1905 144'] = load_one_key_mat(
        basedir / 'Houston2013/HSI.mat',
    )
    lidar: Float64[np.ndarray, '349 1905'] = load_one_key_mat(
        basedir / 'Houston2013/DSM.mat',
    )
    lidar = np.expand_dims(lidar, axis=-1)

    lbl_train :Float64[np.ndarray, '349 1905'] = load_one_key_mat(
        basedir / 'Houston2013/TR.mat',
    )
    lbl_train = coo_array(np.int16(lbl_train))
    lbl_test :Float64[np.ndarray, '349 1905'] = load_one_key_mat(
        basedir / 'Houston2013/TE.mat',
    )
    lbl_test = coo_array(np.int16(lbl_test))

    
    info :DataMetaInfo = {
        'name': 'houston2013-mat',
        'full_name': 'IEEE GRSS DF Contest Houston 2013',
        'homepage': 'https://machinelearning.ee.uh.edu/?page_id=459',
        'n_channel_hsi': 144,
        'n_channel_dsm': 1,
        'n_class': 15,
        'width': 1905,
        'height': 349,
        'label_name': {
            1 : 'Healthy grass',
            2 : 'Stressed grass',
            3 : 'Synthetic grass',
            4 : 'Trees',
            5 : 'Soil',
            6 : 'Water',
            7 : 'Residential',
            8 : 'Commercial',
            9 : 'Road',
            10: 'Highway',
            11: 'Railway',
            12: 'Parking Lot 1',
            13: 'Parking Lot 2',
            14: 'Tennis Court',
            15: 'Running Track'
        },
        "wavelength": np.array([
            364.000000,  368.799988,  373.600006,  378.399994,  383.200012,  387.899994,
            392.700012,  397.500000,  402.299988,  407.000000,  411.799988,  416.600006,
            421.399994,  426.100006,  430.899994,  435.700012,  440.500000,  445.200012,
            450.000000,  454.799988,  459.600006,  464.299988,  469.100006,  473.899994,
            478.600006,  483.399994,  488.200012,  492.899994,  497.700012,  502.500000,
            507.299988,  512.000000,  516.799988,  521.599976,  526.299988,  531.099976,
            535.900024,  540.599976,  545.400024,  550.200012,  554.900024,  559.700012,
            564.500000,  569.200012,  574.000000,  578.799988,  583.500000,  588.299988,
            593.099976,  597.799988,  602.599976,  607.400024,  612.099976,  616.900024,
            621.599976,  626.400024,  631.200012,  635.900024,  640.700012,  645.500000,
            650.200012,  655.000000,  659.799988,  664.500000,  669.299988,  674.099976,
            678.799988,  683.599976,  688.299988,  693.099976,  697.900024,  702.599976,
            707.400024,  712.200012,  716.900024,  721.700012,  726.500000,  731.200012,
            736.000000,  740.700012,  745.500000,  750.299988,  755.000000,  759.799988,
            764.599976,  769.299988,  774.099976,  778.900024,  783.599976,  788.400024,
            793.200012,  797.900024,  802.700012,  807.500000,  812.200012,  817.000000,
            821.799988,  826.500000,  831.299988,  836.099976,  840.799988,  845.599976,
            850.400024,  855.099976,  859.900024,  864.700012,  869.400024,  874.200012,
            879.000000,  883.700012,  888.500000,  893.299988,  898.000000,  902.799988,
            907.599976,  912.299988,  917.099976,  921.900024,  926.700012,  931.400024,
            936.200012,  941.000000,  945.799988,  950.500000,  955.299988,  960.099976,
            964.799988,  969.599976,  974.400024,  979.200012,  983.900024,  988.700012,
            993.500000,  998.299988,  1003.099976, 1007.799988, 1012.599976, 1017.400024,
            1022.200012, 1026.900024, 1031.699951, 1036.500000, 1041.300049, 1046.099976
        ])
    }

    return hsi.transpose(2,0,1), lidar.transpose(2,0,1), lbl_train, lbl_test, info



