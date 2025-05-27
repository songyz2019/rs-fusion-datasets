import numpy as np
from rs_fusion_datasets import fetch_houston2013, fetch_muufl, fetch_trento, hsi2pseudo_rgb
import skimage.io

for loader in [fetch_houston2013, fetch_muufl, fetch_trento]:
    if loader == fetch_houston2013:
        hsi, lidar, _, _, info = loader()
    else:
        hsi, lidar, _, info = loader()
    hsi = hsi.astype(np.float32)
    channel_index_dict = {
        'houston2013': [60, 40, 20],
        'muufl': [20, 15, 5],
        'trento': [40, 20 , 10]
    }
    channel_index = channel_index_dict[info['name']]
    rgb = hsi2pseudo_rgb(hsi, channel_index)
    skimage.io.imsave(f'dist/{info["name"]}_pesudo_rgb.png', rgb.transpose(1, 2, 0))