# SPDX-FileCopyrightText: 2025-present songyz2023 <songyz2023dlut@outlook.com>
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import unittest

import numpy as np
import skimage
from rs_fusion_datasets import fetch_houston2013_mmr, _Houston2013Mmrs, fetch_houston2018
from torch.utils.data import DataLoader
from itertools import product
from hsi2rgb import hsi2rgb
import torch
from typing import get_type_hints, TypedDict, get_origin

from rs_fusion_datasets import CommonHsiDsmDataset, DataMetaInfo
from rs_fusion_datasets.util.fileio import ftp_download


def is_typeddict_instance(obj, typeddict_cls):
    if not isinstance(obj, dict):
        return False
    type_hints = get_type_hints(typeddict_cls)
    for key, expected_type in type_hints.items():
        if key not in obj: # do not check value: isinstance() argument 2 cannot be a parameterized generic
            print(f"Key '{key}' is missing or has incorrect type.")
            return False
    return True

    

class Test(unittest.TestCase):
    def setUp(self):
        pass

    def torch_dataloader_test(self, dataset :CommonHsiDsmDataset):
        b = 16
        dataloader = DataLoader(dataset, batch_size=b, shuffle=True, drop_last=True)
        x_h, x_l, y, extras = next(iter(dataloader))
        
        n_test = 10
        for x_h, x_l, y, extras in dataloader:
            if torch.cuda.is_available():
                x_h, x_l, y = x_h.cuda(), x_l.cuda(), y.cuda()
            self.assertIsInstance(x_h, torch.Tensor)
            self.assertIsInstance(x_l, torch.Tensor)
            self.assertIsInstance(y, torch.Tensor)
            self.assertEqual(x_h.shape, torch.Size([b, dataset.INFO['n_channel_hsi'],dataset.patch_size,dataset.patch_size]))
            self.assertEqual(x_l.shape, torch.Size([b, 1,dataset.patch_size,dataset.patch_size]))
            self.assertEqual(y.shape, torch.Size([b, dataset.n_class]))
            self.assertEqual(x_h.dtype, torch.float)
            self.assertEqual(x_l.dtype, torch.float)
            self.assertEqual(y.dtype, torch.float)
            if n_test <= 0:
                break
            else:
                n_test -= 1

    def generate_lbl2rgb(self, truth, info, subset):
        h,w = truth.shape
        y = np.eye(info['n_class']+1)[truth.todense()].transpose(2, 0, 1) # One Hot
        self.assertEqual(y.shape, (info['n_class']+1, h, w))

        rgb = lbl2rgb(y, info['name'])
        self.assertEqual(rgb.shape, (3, h, w))
        self.assertLessEqual(rgb.max(), 1.0)
        self.assertGreaterEqual(rgb.min(), 0.0)

        img = (rgb*255.0).astype(np.uint8).transpose(1, 2, 0)
        skimage.io.imsave(f"dist/{info['name']}_{subset}.png", img, check_contrast=False)

    def test_fetch_houston2013(self):
        casi, lidar, train_truth, test_truth, info = fetch_houston2013_mmr()
        H, W = 349, 1905
        C_H, C_L = 144, 1
        self.assertEqual(train_truth.data.max(), info['n_class'])
        self.assertEqual(train_truth.data.min(), 1)
        self.assertEqual(train_truth.todense().min(), 0)
        self.assertEqual(test_truth.data.max(), info['n_class'])
        self.assertEqual(test_truth.data.min(), 1)
        self.assertEqual(test_truth.todense().min(), 0)
        self.assertEqual(casi.shape, (C_H, H, W))
        self.assertEqual(lidar.shape, (C_L, H, W))
        self.assertEqual(train_truth.shape, (H, W))
        self.assertEqual(test_truth.shape, (H, W))
        self.assertEqual(info['n_channel_hsi'], 144)
        self.assertEqual(info['n_channel_lidar'], 1)
        self.assertEqual(info['n_class'], 15)
        self.assertEqual(info['width'], W)
        self.assertEqual(info['height'], H)
        self.assertEqual(len(info['label_name']), 15)
        self.assertEqual(info['wavelength'].shape, (144,))
        self.assertEqual(info['wavelength'][0], 364.000000)
        self.assertEqual(info['wavelength'][-1], 1046.099976)
        self.assertEqual(len(info['wavelength']), C_H)
        self.assertTrue(is_typeddict_instance(info, DataMetaInfo))

        hsi = casi.astype(np.float32)
        hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min())
        rgb = hsi2rgb(hsi, wavelength=info['wavelength'], input_format='CHW', output_format='HWC')
        skimage.io.imsave(f"dist/{info['name']}_hsi.png", (rgb * 255.0).astype(np.uint8))

        dsm = lidar[0, :, :]
        dsm_img = (dsm - dsm.min()) / (dsm.max() - dsm.min()) * 255.0
        skimage.io.imsave(f"dist/{info['name']}_dsm.png", dsm_img.astype(np.uint8))

    def test_torch_datasets(self):
        for Dataset, subset, patch_size in product([_Houston2013Mmrs], ['train', 'test', 'full'], [1, 5, 10, 11]):
            dataset = Dataset(subset=subset, patch_size=patch_size)
            self.torch_dataloader_test(dataset)

    def test_fetch_houston2018(self):
        hsi, lidar, lbl, info = fetch_houston2018()
        hsi = hsi.astype(np.float32)
        hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min())
        rgb = hsi2rgb(hsi, wavelength=info['wavelength'], input_format='CHW', output_format='HWC')
        skimage.io.imsave(f"dist/{info['name']}_hsi.png", (rgb * 255.0).astype(np.uint8))

        dsm = lidar[0, :, :]
        dsm_img = (dsm - dsm.min()) / (dsm.max() - dsm.min()) * 255.0
        skimage.io.imsave(f"dist/{info['name']}_dsm.png", dsm_img.astype(np.uint8))


if __name__ == '__main__':
    unittest.main()