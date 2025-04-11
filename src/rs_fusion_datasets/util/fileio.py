import os
from os.path import expanduser, join
from pathlib import Path
from typing import List
import warnings
import hashlib
from io import StringIO
from zipfile import ZipFile

import numpy as np
from scipy.sparse import coo_array
import urllib

def get_data_home(data_home=None) -> str:
    if data_home is None:
        data_home = os.environ.get("SCIKIT_LEARN_DATA", join("~", "scikit_learn_data"))
    data_home = expanduser(data_home)
    os.makedirs(data_home, exist_ok=True)
    return data_home


def read_roi(path :Path, shape) -> coo_array:
    """
    读取ENVI软件导出roi文件得到的txt文件,得到一个稀疏矩阵图像

    用起来像字典

    :param path: 文件路径
    :return: An coo_array image representing the ROI
    """
    warnings.simplefilter("ignore", category=UserWarning) # Supress loadtxt's warning when data is empty

    img = coo_array(shape, dtype='uint')
    buf = ""
    cid = 1
    
    with open(path, 'r') as f:
        for line in f:
            # Comments
            if line.startswith(";") or line.isspace():
                continue

            # Seprator
            if line.lstrip().startswith("1 "): # magick string for compatibility
                data=np.loadtxt(StringIO(buf), usecols=(2, 1), comments=';', dtype='uint')
                buf = ""
                if data.size > 0:
                    rows,cols = data.T
                    vals = cid*np.ones_like(rows)
                    cid += 1
                    # breakpoint()
                    img += coo_array((vals,(rows,cols)), shape=shape) # There may be duplicate points in roi file, so these steps are essential
                    img.data[img.data>cid] = 0  # 清除重复像素点
            # Data
            else:
                buf += line

        # Read last block  
        if buf!="":
            data=np.loadtxt(StringIO(buf), usecols=(2, 1), comments=';', dtype='uint')
            buf = ""
            if data.size > 0:
                rows,cols = data.T
                vals = cid*np.ones_like(rows)
                img += coo_array((vals,(rows,cols)), shape=shape)


    warnings.resetwarnings()

    return img.tocoo()

# TODO: deprecate this
def verify_files(root: Path, files_sha256: dict, extra_message: str = '') -> None:
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


def verify_file(path: Path, expected_sha256: str) -> bool:
    def _quick_sha256(path):
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

    if not path.exists():
        return False
    if not path.is_file():
        raise ValueError(f"{path} is not a file")
    if expected_sha256 != _quick_sha256(path):
        return False
    return True

def zip_download_and_extract(dir_name, url :str, required_files :dict, datahome = None) -> None:
    """
    Download a zip file from url and extract it to datahome

    By calling this, either of the two scenarios will be satisfied:
    1. The program raise an Exception
    2. All files in required_files are found and verified in datahome
    The required_files[dir_name+'.zip'] is the zip file itself, and the rest are the files in the zip file.
    Put your file to basedir/zip_name to skip the download step.
    """
    dir_name = dir_name.removesuffix('/')
    if isinstance(dir_name, str):
        basedir = Path(get_data_home(datahome))/dir_name
        basedir.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f"dir_name is not a valid string or Path: {dir_name}")
        
    
    # Everything is ok, return
    if all([verify_file(basedir/path, sha256) for path,sha256 in required_files.items()]):
        return basedir
    

    zip_name = dir_name+'.zip'
    if not verify_file(basedir/zip_name, required_files[zip_name]):
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36')]
        urllib.request.install_opener(opener)
        print(f"Downloading {url} to {basedir/zip_name}")
        urllib.request.urlretrieve(url, basedir/zip_name, reporthook=lambda blocknum, bs, size: print(f"Downloading {blocknum*bs/1024/1024:0.3f}MB/{size/1024/1024:0.3f}MB", end="\r"))
        urllib.request.install_opener(urllib.request.build_opener()) # Reset
        print() # avoid \r issue

        verify_file(basedir/zip_name, required_files[zip_name])
    # Now the zip file must exist and verified, or the program raised an exception
    # Assuming the zip file is OK

    with ZipFile(basedir/zip_name, 'r') as f_zip:
            f_zip.extractall(basedir)

    for path, sha256 in required_files.items():
        if path == zip_name:
            continue
        if not verify_file(basedir/path, sha256):
            raise ValueError(f"{basedir/path} is not found or not verified")

    return basedir
