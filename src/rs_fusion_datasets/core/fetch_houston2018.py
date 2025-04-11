# SPDX-FileCopyrightText: 2025-present songyz2023 <songyz2023dlut@outlook.com>
# SPDX-License-Identifier: Apache-2.0

import os
from os.path import exists
from pathlib import Path
from typing import Optional
from zipfile import ZipFile
import urllib
import urllib.request
import logging

import numpy as np
import rasterio
from scipy.sparse import coo_array, spmatrix
from jaxtyping import UInt16, Float32, UInt64

from ..util.fileio import zip_download_and_extract
from .common import DataMetaInfo

def fetch_houston2018(datahome: Optional[str] = None, download_if_missing=True) -> tuple[
    UInt16[np.ndarray, '144 349 1905'],
    Float32[np.ndarray, '1 349 1905'],
    UInt64[spmatrix, '349 1905'],
    UInt64[spmatrix, '349 1905'],
    DataMetaInfo
]:
    """Load the Houston2018 data-set in scikit-learn style

    Download it if necessary. All the image are CHW formats. And the shape are typed in the return type.
    The background label is 0, and the rest are 1-15.
    
    :param datahome: The path to store the data files, default is '~/scikit_learn_data'
    :param download_if_missing: Whether to download the data if it is not found

    :return: (hsi, dsm, train_truth, test_truth, info)
    """
    logger = logging.getLogger("fetch_houston2018")

    basedir=zip_download_and_extract('houston2018', 'http://machinelearning.ee.uh.edu/QZ23es1aMPH/2018IEEE/phase2.zip', {
        'houston2018.zip': 'aa6ae39bf9df16b6a72eddaec7e05d853ef525a2f43555f1b9394ec0fe801f19',
        "2018IEEE_Contest/Phase2/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif": "025b3813a0a52855567debf4b1e294937fd9b1ff17e00b9acd9e794ef29696d0",
        "2018IEEE_Contest/Phase2/FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix": "6f29fd1863ceb0cf4ae3b162ffc48e11ce02bad46780531a99ac2b29164b38e2",
        '2018IEEE_Contest/Phase2/Lidar GeoTiff Rasters/DEM+B_C123/UH17_GEM051.tif': '521b04abe70ef1452cdd74ee46b30efa4cc093e707a365bbdd6ad54f6317779a'
    })

    # 3. 数据加载
    with rasterio.open(basedir / '2018IEEE_Contest/Phase2/FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix') as f:
        hsi = f.read() # (50, 1202, 4172)
    with rasterio.open(basedir / '2018IEEE_Contest/Phase2/Lidar GeoTiff Rasters/DEM+B_C123/UH17_GEM051.tif') as f:
        dem = f.read() # (1, 2404, 8344)
    with rasterio.open(basedir / '2018IEEE_Contest/Phase2/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif') as f:
        lbl = f.read() # (1,1202,4768) TODO shape are different

    breakpoint()
    c_h, h, w = hsi.shape

    print(f"{hsi.shape=}, {dem.shape=}, {lbl.shape=}")

    info :DataMetaInfo = {
        'name': 'houston2018',
        'full_name': 'IEEE GRSS DF Contest Houston 2018',
        'homepage': 'https://machinelearning.ee.uh.edu/2018-ieee-grss-data-fusion-challenge-fusion-of-multispectral-lidar-and-hyperspectral-data/',
        'n_channel_hsi': 50,
        'n_channel_lidar': 1,
        'n_class': 20,
        'width': w,
        'height': h,
        'label_name': {
            0: 'Unclassified',
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
        "wavelength": np.linspace(380, 1050, c_h), # TODO: find the real wavelength
    }

    return hsi, dem, lbl, info


__all__ = ['fetch_houston2018']


