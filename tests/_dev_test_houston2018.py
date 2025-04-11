# SPDX-FileCopyrightText: 2025-present songyz2023 <songyz2023dlut@outlook.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import skimage
from fetch_houston2013 import fetch_houston2018
from hsi2rgb import hsi2rgb



class Test(unittest.TestCase):
    def test_fetch_houston2018(self):
        casi, lidar, train_truth, test_truth, info = fetch_houston2018()
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

        

if __name__ == '__main__':
    unittest.main()