# SPDX-FileCopyrightText: 2025-present songyz2023 <songyz2023dlut@outlook.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import skimage
from rs_fusion_datasets import fetch_houston2018
from rs_fusion_datasets.util.fileio import zip_download_and_extract
from hsi2rgb import hsi2rgb



class Test(unittest.TestCase):
    # def test_download(self):
    #     zip_download_and_extract('trento', 'https://github.com/tyust-dayu/Trento/archive/b4afc449ce5d6936ddc04fe267d86f9f35536afd.zip', {
    #         'trento.zip'      :'b203331b039d994015c4137753f15973cb638046532b8dced6064888bf970631',
    #         'Trento-b4afc449ce5d6936ddc04fe267d86f9f35536afd/allgrd.mat'      :'7e3fb2a2ea22c2661dfc768db3cb93c9643b324e7e64fadedfa57f5edbf1818f',
    #         'Trento-b4afc449ce5d6936ddc04fe267d86f9f35536afd/Italy_hsi.mat'   :'7b965fd405314b5c91451042e547a1923be6f5a38c6da83969032cff79729280',
    #         'Trento-b4afc449ce5d6936ddc04fe267d86f9f35536afd/Italy_lidar.mat' :'a04dc90368d6a7b4f9d3936024ba9fef4105456c090daa14fff31b8b79e94ab1',
    #     })

    def test_fetch_houston2018(self):
        hsi, lidar, lbl, info = fetch_houston2018()
        self.assertEqual(lbl.data.max(), info['n_class'])
        self.assertEqual(lbl.data.min(), 1)
        self.assertEqual(lbl.todense().min(), 0)
        self.assertEqual(hsi.shape, (C_H, H, W))
        self.assertEqual(lidar.shape, (C_L, H, W))
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

        hsi = hsi.astype(np.float32)
        hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min())
        rgb = hsi2rgb(hsi, wavelength=info['wavelength'], input_format='CHW', output_format='HWC')
        skimage.io.imsave(f"dist/{info['name']}_hsi.png", (rgb * 255.0).astype(np.uint8))

        dsm = lidar[0, :, :]
        dsm_img = (dsm - dsm.min()) / (dsm.max() - dsm.min()) * 255.0
        skimage.io.imsave(f"dist/{info['name']}_dsm.png", dsm_img.astype(np.uint8))

        

if __name__ == '__main__':
    unittest.main()