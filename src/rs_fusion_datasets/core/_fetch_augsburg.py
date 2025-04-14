from typing import List, Union
import numpy as np
import scipy.io

from scipy.sparse import coo_array
from jaxtyping import Float

from ..util.fileio import zip_download_and_extract
from .common import DataMetaInfo


def fetch_augsburg(url :Union[str, List[str]]='ftp://m1657312@dataserv.ub.tum.de?m1657312/Augsburg_data_4_publication.zip'):
    raise NotImplementedError("This dataset is implemented ")
    """
    Donwload and load the augsburg dataset.

    Use CHW format
    """

    basedir = zip_download_and_extract('augsburg', url, {
        'augsburg.zip'      :'b203331b039d994015c4137753f15973cb638046532b8dced6064888bf970631',
        'augsburg-b4afc449ce5d6936ddc04fe267d86f9f35536afd/allgrd.mat'      :'7e3fb2a2ea22c2661dfc768db3cb93c9643b324e7e64fadedfa57f5edbf1818f',
        'augsburg-b4afc449ce5d6936ddc04fe267d86f9f35536afd/Italy_hsi.mat'   :'7b965fd405314b5c91451042e547a1923be6f5a38c6da83969032cff79729280',
        'augsburg-b4afc449ce5d6936ddc04fe267d86f9f35536afd/Italy_lidar.mat' :'a04dc90368d6a7b4f9d3936024ba9fef4105456c090daa14fff31b8b79e94ab1',
    })


    # 3. 数据加载
    hsi = scipy.io.loadmat(
        basedir / 'augsburg-b4afc449ce5d6936ddc04fe267d86f9f35536afd/Italy_hsi.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['data']

    lidar = scipy.io.loadmat(
        basedir / 'augsburg-b4afc449ce5d6936ddc04fe267d86f9f35536afd/Italy_lidar.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['data']

    truth = scipy.io.loadmat(
        basedir / 'augsburg-b4afc449ce5d6936ddc04fe267d86f9f35536afd/allgrd.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['mask_test']
    truth = coo_array(truth)

    info :DataMetaInfo = {
        'name': 'augsburg',
        'full_name': 'augsburg',
        'version': '0.1',
        'homepage': '',
        'license': '',
        'n_channel_hsi': hsi.shape[-1],
        'n_channel_lidar': lidar.shape[-1],
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



__all__ = ['fetch_augsburg']
