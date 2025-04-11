import os
from os.path import exists, expanduser, join
from pathlib import Path
import warnings
import hashlib
from zipfile import ZipFile
import urllib
import urllib.request
from io import StringIO
import logging
from ftplib import FTP

import numpy as np
from numpy.typing import NDArray

import skimage
from torch.utils.data import Dataset
from scipy.sparse import coo_array


def _get_data_home(data_home=None) -> str:
    if data_home is None:
        data_home = os.environ.get("SCIKIT_LEARN_DATA", join("~", "scikit_learn_data"))
    data_home = expanduser(data_home)
    os.makedirs(data_home, exist_ok=True)
    return data_home



def _download_zip(path: Path, download_if_missing: bool = True):
    if not exists(path):
        if download_if_missing:
            size_mb = 37236640069 // 1048576
            i = 0

            def retr_callback(chunk: bytes):
                global i
                i += 1
                print(f"{i / 128:.2f}MB of {size_mb}MB", end='\r')
                fp.write(chunk)

            ftp = FTP('dataserv.ub.tum.de')
            ftp.login(user='m1657312', passwd='m1657312')

            #with open('checksums.sha512', 'wb') as fp:
            #    ftp.retrbinary('RETR checksums.sha512', fp.write)

            print(f"Downloading Augsburg_data_4_publication.zip ({size_mb} MB)")
            with open(path, 'wb') as fp:
                ftp.retrbinary('RETR Augsburg_data_4_publication.zip', retr_callback, blocksize=8192)
        else:
            raise FileNotFoundError(f"{path} not found")

    _verify_files_sha512(path.parent, {path.name: "34fb35dfdb85ecf247b345cf5852a7314f762d9ce080c6eada1e69415637f3db37fe53eda7987de3c7507955417df6429ecac3edd70215dc12ce695753e6f569"})
    return path



def _verify_files_sha512(root: Path, files_sha512: dict, extra_message: str = '') -> None:
    """验证root下的文件的sha512是否与files_sha256相符

    :param extra_message: 额外报错信息
    :param files_sha512: 例如: `{"1.txt", "f4d619....", "2.txt": "9d03010....."}`
    :param root: 文件夹目录
    """

    def sha512(path):
        """Calculate the sha512 hash of the file at path."""
        sha512hash = hashlib.sha512()
        chunk_size = 8192
        with open(path, "rb") as f:
            while True:
                buffer = f.read(chunk_size)
                if not buffer:
                    break
                sha512hash.update(buffer)
        return sha512hash.hexdigest()

    for filename, checksum in files_sha512.items():
        assert sha512(root/filename) == checksum, f"Incorrect SHA512 for {filename}. Expect {checksum}, Actual {sha512(root/filename)}. {extra_message}"



def _verify_files(root: Path, files_sha256: dict, extra_message: str = '') -> None:
    """验证root下的文件的sha256是否与files_sha256相符

    :param extra_message: 额外报错信息
    :param files_sha256: 例如: `{"1.txt", "f4d619....", "2.txt": "9d03010....."}`
    :param root: 文件夹目录
    """

    def sha256(path):
        """Calculate the sha256 hash of the file at path."""
        sha256hash = hashlib.sha256()
        chunk_size = 8192
        with open(path, "rb") as f:
            while True:
                buffer = f.read(chunk_size)
                if not buffer:
                    break
                sha256hash.update(buffer)
        return sha256hash.hexdigest()

    for filename, checksum in files_sha256.items():
        assert sha256(root/filename) == checksum, f"Incorrect SHA256 for {filename}. Expect {checksum}, Actual {sha256(root/filename)}. {extra_message}"


def fetch_augsburg(datahome=None, download_if_missing=True):
    logger = logging.getLogger("fetch_augsburg")


    # 1. 准备数据
    DATA_HOME = Path(_get_data_home(datahome))
    ZIP_PATH = DATA_HOME / 'Houston2013.zip'
    UNZIPED_PATH = DATA_HOME / 'Houston2013/'
    FILES_PATH = UNZIPED_PATH/'Augsburg_data_4_publication/'
    FILES_SHA256 = {
        "2013_IEEE_GRSS_DF_Contest_CASI.hdr": "869be3459978b535b873bca98b1cf05066c7acca9c160b486a86efd775005e8d",
        "2013_IEEE_GRSS_DF_Contest_CASI.tif": "1440f38594e8e82cc1944c084fc138ef55a70af54122828e999c4fb438574c14",
        "2013_IEEE_GRSS_DF_Contest_LiDAR.hdr": "053c083de1cb0d9ad51c56964b29669733ef2c7db05997d4f4e0779ab2f6aade",
        "2013_IEEE_GRSS_DF_Contest_LiDAR.tif": "9f4facce8876ee84642d9cb03536baf0389506de97ddc01b73366fe4521de981",
        "2013_IEEE_GRSS_DF_Contest_Samples_TR.roi": "feedf41f7064d8f80cf2d9bda72fcbcc98b64658d01e519ad0b90b1ca88f1375",
        "2013_IEEE_GRSS_DF_Contest_Samples_TR.txt": "16c69cf216535d7b4df2045b05d49c50a078609aa6d011a5e23e54f4cd08abda",
        "2013_IEEE_GRSS_DF_Contest_Samples_VA.zip": "aac7015c7a986063002a86eb7f7cc57ed6f14f5eaf3e9ca29c0cb1e63fd7e0d5",
        "copyright.txt": "63d908383566b1ff6fd259aa202e31dab9a629808919d87d94970df7ad25180d",
        "2013_IEEE_GRSS_DF_Contest_Samples_VA.txt": "768bb02193d04c8020b45f1f31a49926a5b914040f77f71a81df756d6e8b8dcb"
    }
    if not exists(DATA_HOME):
        os.makedirs(DATA_HOME)

    # 2.下载数据集zip文件并解压
    if exists(FILES_PATH) and len(os.listdir(FILES_PATH)) > 0:  # 已存在解压的文件夹且非空
        _verify_files(FILES_PATH, FILES_SHA256, f"please try removing {FILES_PATH}")
    else:
        _download_zip(ZIP_PATH, download_if_missing)
        # 解压 2013_DFTC 目录下的所有文件
        logger.info(f"Decompressing {ZIP_PATH}")
        with ZipFile(ZIP_PATH, 'r') as zip_file:
            zip_file.extractall(UNZIPED_PATH)

        # 删除ZIP
        os.remove(ZIP_PATH)


    # 3. 数据加载
    lidar :NDArray = skimage.io.imread(FILES_PATH / 'HySpex.tif')[np.newaxis, :, :] # (1   349 1905)
    casi  :NDArray = skimage.io.imread(FILES_PATH / '3K_DSM.tif' ).transpose(2,0,1)  # (144 349 1905)

    info = {
        'name': 'Augsburg (MDAS)',
        'homepage': 'https://mediatum.ub.tum.de/1657312',
        'license': 'CC BY-SA',
        'modalities': ['hsi','dsm','sar','gis','msi'],
        'n_band_hsi': 144,
        'n_band_lidar': 1,
        'n_class': 15,
        'width': 1905,
        'height': 349,
        'label_dict': {
            1: 'Forest',
            2: 'Park',
            3: 'Residential',
            4: 'Industrial',
            5: 'Farm',
            6: 'Cemetery',
            7: 'Allotments',
            8: 'Meadow',
            9: 'Commercial',
            10: 'Recreation ground',
            11: 'Retail',
            12: 'Scrub',
            13: 'Grass',
            14: 'Heath'
        },
        "wavelength": np.array([

        ])
    }

    return casi, lidar, 0, 0, info



__all__ = []