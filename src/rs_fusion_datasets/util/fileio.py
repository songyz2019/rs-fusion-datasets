from ftplib import FTP
import logging
import os
from pathlib import Path
import re
from typing import List, Union, Optional, Tuple, Dict
import warnings
import hashlib
from io import StringIO
from zipfile import ZipFile

import numpy as np
import scipy
from scipy.sparse import coo_array, sparray
from jaxtyping import UInt16
import urllib


def get_data_home(data_home :Optional[Union[Path, str]]=None) -> Path:
    if data_home is None:
        data_home = os.environ.get("SCIKIT_LEARN_DATA", os.path.join("~", "scikit_learn_data")) # Not neat but works
        data_home = Path(data_home)
    elif isinstance(data_home, str):
        data_home = Path(data_home)
    data_home = data_home.expanduser()
    data_home.mkdir(parents=True, exist_ok=True)
    return data_home


def read_roi(path :Path, shape :Tuple[int, int]) -> UInt16[sparray, 'h w']:
    """
    读取ENVI软件导出roi文件得到的txt文件,得到一个稀疏矩阵图像

    用起来像字典

    :param path: 文件路径
    :return: An coo_array image representing the ROI
    """
    warnings.simplefilter("ignore", category=UserWarning) # Supress loadtxt's warning when data is empty

    img = coo_array(shape, dtype='uint16')
    buf = ""
    cid = 1
    
    with open(path, 'r') as f:
        for line in f:
            # Comments
            if line.startswith(";") or line.isspace():
                continue

            # Seprator
            if line.lstrip().startswith("1 "): # magick string for compatibility
                data=np.loadtxt(StringIO(buf), usecols=(2, 1), comments=';', dtype='uint16', converters=float)
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
            data=np.loadtxt(StringIO(buf), usecols=(2, 1), comments=';', dtype='uint16')
            buf = ""
            if data.size > 0:
                rows,cols = data.T
                vals = cid*np.ones_like(rows)
                img += coo_array((vals,(rows,cols)), shape=shape)


    warnings.resetwarnings()

    return img.tocoo()

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

    logging.debug(f"Verifying {path} with sha256 {expected_sha256}")
    if not path.exists():
        logging.debug(f"Not exist")
        return False
    if not path.is_file():
        logging.debug(f"Not a file")
        raise ValueError(f"{path} is not a file")
    if expected_sha256 != _quick_sha256(path):
        logging.debug(f"Incorrect sha256: {_quick_sha256(path)}")
        return False
    logging.debug(f"PASS")
    return True

def zip_download_and_extract(dir_name :str, url :Union[str, List[str]], required_files :Dict[str, str], data_home :Optional[Union[str,Path]] = None) -> None:
    """
    Download a zip file from url and extract it to datahome

    By calling this, either of the two scenarios will be satisfied:
    1. The program raise an Exception
    2. All files in required_files are found and verified in datahome
    The required_files[dir_name+'.zip'] is the zip file itself, and the rest are the files in the zip file.
    Put your file to basedir/zip_name to skip the download step.
    """
    # breakpoint()
    dir_name = dir_name.removesuffix('/')
    if isinstance(dir_name, str):
        basedir = Path(get_data_home(data_home))/dir_name
        basedir.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f"dir_name is not a valid string or Path: {dir_name}")
        
    
    # Everything is ok, return
    if all([verify_file(basedir/path, sha256) for path,sha256 in required_files.items()]):
        return basedir
    

    zip_name = dir_name+'.zip'
    mirrored_download(
        basedir/zip_name,
        url,
        required_files[zip_name]
    )

    # Now the zip file must exist and verified, or the program raised an exception
    # Assuming the zip file is OK

    with ZipFile(basedir/zip_name, 'r') as f_zip:
            f_zip.extractall(basedir)

    for path, sha256 in required_files.items():
        if path == zip_name:
            continue
        if not verify_file(basedir/path, sha256):
            raise ValueError(f"{basedir/path} is not found or not verified. If it exists, please delete {basedir} and try again.")

    return basedir

def _ftp_download(path :Path, url, sha256: str) -> None:
    """Only simple case url supported ftp://username@host.com?password/aug.zip"""
    grp = re.match(r'ftp://(\w+)@([\w\.]+)\?([\w\.]+)/([\w\.]+)', url)
    ftp_username, ftp_host, ftp_password, ftp_file_name = grp.groups()

    if verify_file(path, sha256):
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    ftp = FTP(ftp_host)
    ftp.login(user=ftp_username, passwd=ftp_password)
    ftp.voidcmd('TYPE I')

    size_mb = ftp.size(ftp_file_name) / 1024 / 1024
    with open(path, 'wb') as fp:
        global i_rsfds
        i_rsfds = 0 # add a random suffix
        def retr_callback(chunk: bytes):
            global i_rsfds
            i_rsfds += len(chunk)
            print(f"{i_rsfds/1024/1024:.3f}MB/{size_mb:.3f}MB", end='\r')
            fp.write(chunk)
        ftp.retrbinary(f'RETR {ftp_file_name}', retr_callback, blocksize=65536)

    assert verify_file(path, sha256)





def mirrored_download(path :Path, url :Union[str, List[str]], sha256: str) -> None:
    """Download a file from urls, and check the sha256 hash. If the download fails, try the next url.

    :param path: The path to save the file
    :param urls: The list of urls to download from
    :param sha256: The sha256 hash of the file
    """
    if verify_file(path, sha256):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(url, str):
        url = [url,]
    
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0')]
    urllib.request.install_opener(opener)

    success = False
    for url in url:
        try:
            print(f"Downloading {url} to {path}")
            if url.startswith('http://') or url.startswith('https://'):
                urllib.request.urlretrieve(url, path, reporthook=lambda blocknum, bs, size: print(f"{blocknum*bs/1024/1024:0.3f}MB/{size/1024/1024:0.3f}MB", end="\r")  if blocknum%128==0 else None )
            elif url.startswith('ftp://'):
                _ftp_download(path, url, sha256)
            print("\nDownload Success")
            if verify_file(path, sha256):
                success = True
                break
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
            print(f"Error downloading {url}: {e}")
            continue

    urllib.request.install_opener(urllib.request.build_opener()) # Reset
    if not success:
        raise ValueError(f"Failed to download {path} from all urls")
    else:
        return


def load_one_key_mat(path :Path, *args, **kwargs):
    mat = scipy.io.loadmat(path, squeeze_me=True, mat_dtype=True, struct_as_record=False, *args, **kwargs)
    keys = [x for x in mat.keys() if not x.startswith('__') and not x.endswith('__')]
    assert len(keys) == 1, f"Mat file {path} has more than one key: {keys}, please load it manually"
    return mat[keys[0]]
