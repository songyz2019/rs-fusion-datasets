import os
from os.path import exists
from pathlib import Path
from zipfile import ZipFile
import urllib
import urllib.request
import logging

import numpy as np
import scipy.io

from scipy.sparse import coo_array
from jaxtyping import Float
from ..util.fileio import get_data_home, verify_files, read_roi, zip_download_and_extract
from .common import DataMetaInfo


def fetch_muufl(datahome=None, download_if_missing=True):
    """
    Donwload and load the MUUFL Gulfport dataset.

    Use CHW format
    """

    basedir = zip_download_and_extract('muufl', 'https://github.com/GatorSense/MUUFLGulfport/archive/refs/tags/v0.1.zip', {
        'muufl.zip'      :'b203331b039d994015c4137753f15973cb638046532b8dced6064888bf970631',
        "MUUFLGulfportSceneLabels/muufl_gulfport_campus_1_hsi_220_label.mat": "69420a72866dff4a858ae503e6e2981af46f406a4ad8f4dd642efa43feb59edc"
    })

    # 3. 数据加载
    d = scipy.io.loadmat(
        basedir / 'MUUFLGulfportSceneLabels' / 'muufl_gulfport_campus_1_hsi_220_label.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['hsi']
    hsi = d.Data # HWC
    lidar = d.Lidar[0].z
    truth = d.sceneLabels.labels
    truth[truth==-1] = 0
    truth = coo_array(truth, dtype='int')

    info :DataMetaInfo = {
        'name': 'muufl',
        'full_name': 'MUUFL Gulfport dataset',
        'version': '0.1',
        'homepage': 'https://github.com/GatorSense/MUUFLGulfport',
        'license': 'MIT',
        'n_channel_hsi': hsi.shape[-1],
        'n_channel_lidar': lidar.shape[-1],
        'n_class': d.sceneLabels.Materials_Type.size,
        'width': hsi.shape[1],
        'height': hsi.shape[0],
        'label_name': dict(enumerate(d.sceneLabels.Materials_Type, start=1)),
        'wavelength': np.linspace(350, 1000, hsi.shape[-1]), # TODO: find the real wavelength
    }

    return hsi.transpose(2,0,1), lidar.transpose(2,0,1), truth, info




__all__ = ['fetch_muufl', 'split_spmatrix']
