from pathlib import Path
from typing import List, Union, Optional, Tuple

from numpy import ndarray
import scipy.io

from scipy.sparse import coo_array
from ..util.fileio import zip_download_and_extract
from .common import DataMetaInfo
from jaxtyping import  UInt16, Float32, UInt8, Float64



def fetch_muufl(
        url :Union[str, List[str]]='https://github.com/GatorSense/MUUFLGulfport/archive/refs/tags/v0.1.zip', 
        data_home:Optional[Union[Path, str]]=None
        ) -> Tuple[
        Float32[ndarray, '64 325 220'], 
        Float32[ndarray, '2 325 220'],
        Float64[coo_array, '325 220'],
        DataMetaInfo
    ]:
    """Fetch and load the Muufl dataset.

    The background is 0, and the labels start from 1. All images are CHW formats.
    
    :param url: The URL to download the dataset. Use a list to specify multiple mirrored URLs.
    :param data_home: The path to store the data files, default is SCIKIT_LEARN_DATA environment variable or '~/scikit_learn_data'
    :return: (hsi, lidar, train_labels, train_labels, info)
    """

    basedir = zip_download_and_extract('muufl', url, {
        'muufl.zip'      :'2219e6259e3ad80521a8a7ff879916624efa61eb6df1bfd80538f6f2d4befa2c',
        "MUUFLGulfport-0.1/MUUFLGulfportSceneLabels/muufl_gulfport_campus_1_hsi_220_label.mat": "69420a72866dff4a858ae503e6e2981af46f406a4ad8f4dd642efa43feb59edc"
    }, data_home=data_home)

    d = scipy.io.loadmat(
        basedir / 'MUUFLGulfport-0.1/MUUFLGulfportSceneLabels/muufl_gulfport_campus_1_hsi_220_label.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['hsi']
    hsi :Float32[ndarray, '325 220 64'] = d.Data
    lidar :Float32[ndarray, '325 220 2'] = d.Lidar[0].z
    truth :Float64[ndarray, '325 220'] = d.sceneLabels.labels
    truth[truth==-1] = 0
    truth = coo_array(truth, dtype='int')
    info :DataMetaInfo = {
        'name': 'muufl',
        'full_name': 'MUUFL Gulfport dataset',
        'version': '0.1',
        'homepage': 'https://github.com/GatorSense/MUUFLGulfport',
        'license': 'MIT',
        'n_channel_hsi': hsi.shape[-1],
        'n_channel_dsm': lidar.shape[-1],
        'n_class': d.sceneLabels.Materials_Type.size,
        'width': hsi.shape[1],
        'height': hsi.shape[0],
        'label_name': dict(enumerate(d.sceneLabels.Materials_Type, start=1)),
        'wavelength': d.info.wavelength
    }

    return hsi.transpose(2,0,1), lidar.transpose(2,0,1), truth, info




__all__ = ['fetch_muufl', 'split_spmatrix']
