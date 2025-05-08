from pathlib import Path
from typing import List, Union, Optional, Tuple

from numpy import ndarray
import scipy.io

from scipy.sparse import coo_array
from ..util.fileio import zip_download_and_extract
from .common import DataMetaInfo
from jaxtyping import  UInt16, Float32, UInt8, Float64



def _fetch_muufl_mat(
        url :Union[str, List[str]]='http://10.7.36.2:5000/dataset/muufl_mat.zip', 
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

    basedir = zip_download_and_extract('muufl-mat', url, {
        'muufl-mat.zip'       : '95cd1235a5f02df69cd384de5fb83d0d14fa37ac5776081caf9d477368e9a291',
        'muufl_mat/HSI.mat'   : 'adf5934333753b2e05cf244392b43d86b432aa8dac039964ce1f409c96c65edd',
        'muufl_mat/labels.mat': 'a114cec5f9b51c91fac3041426a7142e6dbc0e6d895af9b3b5cf7819e7c415b0',
        'muufl_mat/Lidar.mat' : '31a9f7d6b2e9ab6507304c72b7b99ec975565fd5961ea9523827443690bb7194'
    }, data_home=data_home)

    hsi :Float32[ndarray, '325 220 64'] = scipy.io.loadmat(
        basedir / 'muufl_mat/HSI.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['HSI']
    lidar :Float32[ndarray, '325 220 2'] = scipy.io.loadmat(
        basedir / 'muufl_mat/Lidar.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['Lidar']
    truth :Float64[ndarray, '325 220'] = d.sceneLabels.labels
    truth[truth==-1] = 0
    truth = coo_array(truth, dtype='int')
    info :DataMetaInfo = {
        'name': 'muufl-mat',
        'full_name': 'MUUFL Gulfport dataset, a mat file that can not find where it comes, only for internal use',
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
