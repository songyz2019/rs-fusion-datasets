from colorsys import hsv_to_rgb
from typing import Union
import skimage
import torch
import numpy as np
from jaxtyping import Float
import warnings

def hue_platte(n_sample :int):
    return [hsv_to_rgb(n/n_sample, 1, 1) for n in range(n_sample)]

def lbl2rgb(lbl :Float[Union[np.ndarray,torch.Tensor], 'C H W'], palette, kind='overlay') -> Float[Union[np.ndarray,torch.Tensor], '... 3 H W']:
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
    placeholder_palette = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
        "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
        "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff", "#000000", "#1f77b4", "#ff7f0e",
        "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    palette_presets = {
        'houston2013': ('forestgreen', 'limegreen', 'darkgreen', 'green', 'indianred', 'royalblue', 'papayawhip', 'pink','red', 'orangered', 'cadetblue', 'yellow', 'darkorange', 'darkmagenta', 'cyan'),
        'houston2013-mmr': ('forestgreen', 'limegreen', 'darkgreen', 'green', 'indianred', 'royalblue', 'papayawhip', 'pink','red', 'orangered', 'cadetblue', 'yellow', 'darkorange', 'darkmagenta', 'cyan'),
        'houston2013-mat': ('forestgreen', 'limegreen', 'darkgreen', 'green', 'indianred', 'royalblue', 'papayawhip', 'pink','red', 'orangered', 'cadetblue', 'yellow', 'darkorange', 'darkmagenta', 'cyan'),
        'muufl': ('forestgreen', 'limegreen', 'lightblue', 'papayawhip', 'red', 'blue', 'purple', 'pink','orangered', 'yellow', 'brown'),
        'muufl-mat': ('forestgreen', 'limegreen', 'lightblue', 'papayawhip', 'red', 'blue', 'purple', 'pink','orangered', 'yellow', 'brown'),
        'trento': ('royalblue','lightblue' , 'limegreen', 'yellow', 'red', 'brown'),
        'trento-mat': ('royalblue','lightblue' , 'limegreen', 'yellow', 'red', 'brown'),
        'houston2018-ouc': placeholder_palette, #TODO
        'augsburg-ouc':    placeholder_palette, #TODO
        'berlin-ouc':      placeholder_palette, #TODO
    }
    if palette in palette_presets:
        palette = palette_presets[palette]
    elif not isinstance(palette, tuple):
        warnings.warn(f"palette not in preset, using default palette")
        palette = placeholder_palette
    
    if isinstance(lbl, torch.Tensor):
        lbl = lbl.cpu().numpy()
    if len(lbl.shape) == 3:
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