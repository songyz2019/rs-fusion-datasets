# SPDX-FileCopyrightText: 2025-present songyz2023 <songyz2023dlut@outlook.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import skimage
from rs_fusion_datasets import fetch_augsburg_ouc, fetch_berlin_ouc, fetch_houston2018_ouc
from rs_fusion_datasets import fetch_houston2013, fetch_muufl, split_spmatrix, fetch_trento, Muufl, Houston2013, Houston2018Ouc, Trento, DataMetaInfo, lbl2rgb, AugsburgOuc, BerlinOuc, Houston2013Mmr
import torch
from torch.utils.data import DataLoader
from itertools import product
from hsi2rgb import hsi2rgb
from rs_fusion_datasets import * # test __all__ in __init__.py

from typing import get_type_hints


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
        import logging
        # logging.basicConfig(level=logging.DEBUG)
        self.trento = fetch_trento()
        self.houston2013 = fetch_houston2013()
        self.muufl = fetch_muufl()
    
    def test_lbl2rgb(self):
        for datafetch in [fetch_augsburg_ouc, fetch_berlin_ouc, fetch_houston2018_ouc, fetch_houston2013, fetch_muufl, fetch_trento]:
            r = datafetch()
            if len(r) == 6:
                hsi, dsm, train_label, test_label, all_label,info = r
            elif len(r) == 5:
                hsi, dsm, train_label, test_label ,info = r
            elif len(r) == 4:
                hsi, dsm, label, info = r
                train_label, test_label = split_spmatrix(label, 100)
            else:
                raise ValueError(f"fetch function {datafetch} should return 4 or 5 or 6 elements, but got {len(r)}")
            self.generate_lbl2rgb(train_label, info, split='train')
            self.generate_lbl2rgb(test_label, info, split='test')



    def torch_dataloader_test(self, dataset):
        rgb = hsi2rgb(dataset.hsi, wavelength=dataset.INFO['wavelength'], input_format='CHW', output_format='HWC')
        skimage.io.imsave(f"dist/torch_{dataset.INFO['name']}_hsi.png", (rgb * 255.0).numpy().astype(np.uint8))
        for i,dsm in enumerate(dataset.dsm):
            dsm *= 255.0
            skimage.io.imsave(f"dist/torch_{dataset.INFO['name']}_dsm{i}.png", dsm.numpy().astype(np.uint8))

        b = 8
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
            self.assertEqual(x_l.shape, torch.Size([b, dataset.INFO['n_channel_dsm'],dataset.patch_size,dataset.patch_size]))
            self.assertEqual(y.shape, torch.Size([b, dataset.n_class]))
            self.assertEqual(x_h.dtype, torch.float)
            self.assertEqual(x_l.dtype, torch.float)
            self.assertEqual(y.dtype, torch.float)
            if n_test <= 0:
                break
            else:
                n_test -= 1

    def generate_lbl2rgb(self, truth, info, split):
        h,w = truth.shape
        y = np.eye(info['n_class']+1)[truth.todense()].transpose(2, 0, 1) # One Hot
        self.assertEqual(y.shape, (info['n_class']+1, h, w))

        rgb = lbl2rgb(y, info['name'])
        self.assertEqual(rgb.shape, (3, h, w))
        self.assertLessEqual(rgb.max(), 1.0)
        self.assertGreaterEqual(rgb.min(), 0.0)

        img = (rgb*255.0).astype(np.uint8).transpose(1, 2, 0)
        skimage.io.imsave(f"dist/{info['name']}_{split}.png", img, check_contrast=False)

    def test_fetch_houston2013(self):
        casi, lidar, train_truth, test_truth, info = self.houston2013
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
        self.assertEqual(info['n_channel_dsm'], 1)
        self.assertEqual(info['n_class'], 15)
        self.assertEqual(info['width'], W)
        self.assertEqual(info['height'], H)
        self.assertEqual(len(info['label_name']), 15)
        self.assertEqual(info['wavelength'].shape, (144,))
        self.assertEqual(info['wavelength'][0], 364.000000)
        self.assertEqual(info['wavelength'][-1], 1046.099976)
        self.assertEqual(len(info['wavelength']), C_H)
        self.assertTrue(is_typeddict_instance(info, DataMetaInfo))



    def test_fetch_muufl(self):
        casi, lidar, truth, info = self.muufl
        train_label, test_label = split_spmatrix(truth, 20)
        H, W = 325, 220
        C_H, C_L = 64, 2
        self.assertEqual(casi.shape, (C_H, H, W))
        self.assertEqual(lidar.shape, (C_L, H, W))
        self.assertEqual(truth.shape, (H, W))
        self.assertEqual(train_label.shape, (H, W))
        self.assertEqual(test_label.shape, (H, W))
        self.assertEqual(info['n_channel_dsm'], C_L)
        self.assertEqual(info['n_channel_hsi'], C_H)
        self.assertEqual(len(info['wavelength']), C_H)
        self.assertTrue(is_typeddict_instance(info, DataMetaInfo))

    def test_fetch_trento(self):
        casi, lidar, truth, info = self.trento
        train_label, test_label = split_spmatrix(truth, 20)

        H, W = 166, 600
        C_H, C_L = 63, 2
        self.assertEqual(casi.shape, (C_H, H, W))
        self.assertEqual(lidar.shape, (C_L, H, W))
        self.assertEqual(info['n_channel_hsi'], C_H)
        self.assertEqual(len(info['wavelength']), C_H)
        self.assertEqual(info['n_channel_dsm'], C_L)

        self.assertEqual(truth.shape, (H, W))
        self.assertEqual(train_label.shape, (H, W))
        self.assertEqual(test_label.shape, (H, W))
        self.assertTrue(is_typeddict_instance(info, DataMetaInfo))

    def test_torch_datasets(self):
        for Dataset, split, patch_size in product([Houston2018Ouc, BerlinOuc,AugsburgOuc,Houston2013, Muufl, Trento, Houston2013Mmr], ['train', 'test', 'full'], [1, 6, 9]):
            dataset = Dataset(split=split, patch_size=patch_size)
            self.torch_dataloader_test(dataset)
            if Dataset in [Muufl, Trento] and split == 'train':
                n_train_perclass = 50
                self.assertEqual(dataset.n_class*n_train_perclass, len(Dataset(split=split, patch_size=5, n_train_perclass=n_train_perclass)))

    def test_datahome(self):
        fetch_trento(data_home='./tmp/')
        Trento(split='train', patch_size=5, root='./tmp/')
                    

if __name__ == '__main__':
    unittest.main()