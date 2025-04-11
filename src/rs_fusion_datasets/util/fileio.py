import os
from os.path import expanduser, join
from pathlib import Path
import warnings
import hashlib
from io import StringIO

import numpy as np
from scipy.sparse import coo_array

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
