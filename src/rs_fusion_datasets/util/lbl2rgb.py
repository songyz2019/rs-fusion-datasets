from typing import Union
import skimage
import torch
import numpy as np
from jaxtyping import Float

def lbl2rgb(lbl :Float[Union[np.ndarray,torch.Tensor], '... C H W'], palette, kind='overlay') -> Float[Union[np.ndarray,torch.Tensor], '... 3 H W']:
    """
    符合实验室内部 格式要求的OneHot标签转图像函数

    :param kind: 分割模式,如果为'overlay'则是硬分割, 'avg'则是软分割
    :param lbl B C H W格式的图像batch, C为OneHot编码格式
    :return: B 3 H W格式的RGB图像, 取值范围为0~1
    """
    def hex2rgb(x :str):
        if not x.startswith("#"):
            return x
        x = x.removeprefix("#")
        r = int(x[:2],  base=16) / 255.0
        g = int(x[2:4], base=16) / 255.0
        b = int(x[4:6], base=16) / 255.0
        return [r,g,b]
    
    palette_presets = {
        'houston2013': ('forestgreen', 'limegreen', 'darkgreen', 'green', 'indianred', 'royalblue', 'papayawhip', 'pink','red', 'orangered', 'cadetblue', 'yellow', 'darkorange', 'darkmagenta', 'cyan'),
        'muufl': ('forestgreen', 'limegreen', 'lightblue', 'papayawhip', 'red', 'blue', 'purple', 'pink','orangered', 'yellow', 'brown'),
        'trento': ('royalblue','lightblue' , 'limegreen', 'yellow', 'red', 'brown')
    }
    if palette in palette_presets:
        palette = palette_presets[palette]

    if len(lbl.shape)==3:
        if isinstance(lbl, torch.Tensor):
            lbl = torch.argmax(lbl, dim=-3).cpu().numpy()
        else:
            lbl = np.argmax(lbl, axis=-3)
    img = skimage.color.label2rgb(
        lbl,
        channel_axis=-3,
        colors=[hex2rgb(x) for x in palette],
        bg_label=0,
        bg_color=hex2rgb('#000000'),
        kind=kind # 硬分割还是软分割
    )
    return img