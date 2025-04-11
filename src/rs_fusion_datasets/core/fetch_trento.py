import os
from os.path import exists, expanduser, join
from pathlib import Path
import hashlib
from zipfile import ZipFile
import urllib
import urllib.request
import logging

import numpy as np
import scipy.io

from scipy.sparse import coo_array
from jaxtyping import Float

from ..util.fileio import get_data_home, verify_files
from .common import DataMetaInfo


def fetch_trento(datahome=None, download_if_missing=True):
    """
    Donwload and load the Trento dataset.

    Use CHW format
    """

    def fetch_zip(url, path: Path, download_if_missing: bool = True) -> Path:
        """Make sure `path` is the zip file of Houston2013, or raise FileNotFoundError
        """
        if not exists(path):
            if download_if_missing:
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36')]
                urllib.request.install_opener(opener)
                logger.info(f"Downloading {url}")
                urllib.request.urlretrieve(url, path)
            else:
                raise FileNotFoundError(f"{path} not found")

        verify_files(path.parent, {path.name: "B203331B039D994015C4137753F15973CB638046532B8DCED6064888BF970631".lower()})
        return path

    # 1. 准备
    logger = logging.getLogger("fetch_trento")
    URL = "https://github.com/tyust-dayu/Trento/archive/b4afc449ce5d6936ddc04fe267d86f9f35536afd.zip"
    DATA_HOME = Path(get_data_home(datahome))
    ZIP_PATH = DATA_HOME / 'Trento-b4afc449ce5d6936ddc04fe267d86f9f35536afd.zip'
    UNZIPED_PATH = DATA_HOME / 'MUUFLGulfport/'
    ROOT = UNZIPED_PATH/'Trento-b4afc449ce5d6936ddc04fe267d86f9f35536afd'
    FILES_SHA256 = {
        "allgrd.mat": "7e3fb2a2ea22c2661dfc768db3cb93c9643b324e7e64fadedfa57f5edbf1818f",
        "Italy_hsi.mat": "7b965fd405314b5c91451042e547a1923be6f5a38c6da83969032cff79729280",
        "Italy_lidar.mat": "a04dc90368d6a7b4f9d3936024ba9fef4105456c090daa14fff31b8b79e94ab1"
    }
    if not exists(DATA_HOME):
        os.makedirs(DATA_HOME)

    # 2.下载数据集zip文件并解压
    if exists(ROOT) and len(os.listdir(ROOT)) > 0:  # 已存在解压的文件夹且非空
        verify_files(ROOT, FILES_SHA256, f"please try removing {ROOT}")
    else:
        fetch_zip(URL, ZIP_PATH, download_if_missing)
        logger.info(f"Decompressing {ZIP_PATH}")
        with ZipFile(ZIP_PATH, 'r') as zip_file:
            zip_file.extractall(UNZIPED_PATH)

        verify_files(ROOT, FILES_SHA256)

        # 删除ZIP
        os.remove(ZIP_PATH)

    # 3. 数据加载
    hsi = scipy.io.loadmat(
        ROOT / 'Italy_hsi.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['data']

    lidar = scipy.io.loadmat(
        ROOT / 'Italy_lidar.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['data']

    truth = scipy.io.loadmat(
        ROOT / 'allgrd.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['mask_test']
    truth = coo_array(truth)

    info :DataMetaInfo = {
        'name': 'trento',
        'full_name': 'Trento',
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
        'wavelength': np.linspace(350, 1000, hsi.shape[-1]), # TODO: find the real wavelength
    }

    return hsi.transpose(2,0,1), lidar.transpose(2,0,1), truth, info



__all__ = ['fetch_trento']
