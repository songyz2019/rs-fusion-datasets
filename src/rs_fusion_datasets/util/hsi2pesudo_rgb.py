from typing import List, Optional, Tuple
from jaxtyping import UInt8, Float
import numpy as np

def hsi2pseudo_rgb(hsi: Float[np.ndarray, 'c h w'], rgb_channel_index: Optional[List[int]]) -> UInt8[np.ndarray, '3 h w']:
    if rgb_channel_index is None:
        n_channel = hsi.shape[0]
        step = n_channel // 7
        rgb_channel_index = [
            step * 2,
            step * 4,
            step * 6
        ]

    hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min())
    if isinstance(rgb_channel_index, tuple):
        rgb_channel_index = list(rgb_channel_index)
    assert len(rgb_channel_index) == 3, f"rgb_channel_index should be a list of 3 integers, but got {len(rgb_channel_index)}"
    rgb = hsi[rgb_channel_index].astype(np.float32)
    rgb *= 255.0
    return rgb.astype(np.uint8)