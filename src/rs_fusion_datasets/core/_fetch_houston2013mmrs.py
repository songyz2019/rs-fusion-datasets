import os
from os.path import exists, expanduser, join
from pathlib import Path
from types import SimpleNamespace
from typing import List
from zipfile import ZipFile
import logging

import numpy as np
from scipy.io import loadmat

import skimage
from scipy.sparse import coo_array

from .common import DataMetaInfo
from ..util.fileio import zip_download_and_extract



def _fetch_houston2013mmrs(datahome=None):
    """

    Image array format: CHW
    You are not supposed to use this function because the Houston2013_mmcr dataset is not public available in a direct link, and you should host
    your dataset and provide the link. The default link is only for internal test. 
    """
    # 1. 准备

    basedir = zip_download_and_extract('houston2013mmcr', 'http://localhost:5000/houston2013mmcr.zip', {
        'houston2013mmcr.zip': '360a56f013479fcb7c46a725adfc44a953d4c4675da10016947559e8005df829'
        'gt.mat' : '75ecccc08ac7709e48285bb098fda802da6efd6dc0168cb1c99c6ce09d0b6ae0',
        'HSI.mat' : '6a0edba3c224df411623ed5774fc34e91929ab341709859b2f56cc38dbb3c6fd',
        'LiDAR.mat' : '7aa956e7c371fd29a495f0cb9bb8f572aaa4065fcfeda2b3e854a5cef74b35ad',
        'TRLabel.mat' : '96ce863eaf4dc548c3140a480dee33c812d46194ae5ed345fed6e71a3d72b527',
        'TSLabel.mat' : '46bd849d556c80ed67b33f23dd288eafa7ac9f97a847390be373b702b0bf5a45'
    })


    # 3.加载数据
    hsi = loadmat(str(basedir / 'HSI.mat'))['HSI'].transpose(2,0,1)
    lidar = loadmat(str(basedir / 'LiDAR.mat'))['LiDAR'] [np.newaxis,:,:]
    tr = loadmat(str(basedir / 'TRLabel.mat'))['TRLabel']
    te = loadmat(str(basedir / 'TSLabel.mat'))['TSLabel']
    gt = loadmat(str(basedir / 'gt.mat'))['gt']

    tr = coo_array(tr, dtype='int')
    te = coo_array(te, dtype='int')

    info :DataMetaInfo = {
        'name': 'houston2013mmrs',
        'full_name': 'MMRS version of IEEE GRSS DF Contest Houston 2013',
        'homepage': 'https://github.com/likyoo/Multimodal-Remote-Sensing-Toolkit',
        'n_channel_hsi': 144,
        'n_channel_lidar': 1,
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
    return hsi, lidar, tr, te, info
