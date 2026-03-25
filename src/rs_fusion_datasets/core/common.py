from typing import Optional, TypedDict
import numpy as np
# It's ugly, but it's for compatibility
try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

DEFAULT_PALETTE  = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
    "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff", "#000000", "#1f77b4", "#ff7f0e",
    "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

class DataMetaInfo(TypedDict):
    """
    DataMetaInfo is a TypedDict that defines the structure for storing metadata 
    and configuration information.

    Attributes:
        name (str): The short name (aka. dataset string id) of the dataset.
        full_name (str): The full descriptive name of the dataset.
        homepage (str): The URL of the dataset's homepage or source.
        n_channel_hsi (int): The number of hyperspectral channels in the dataset.
        n_channel_lidar (int): The number of LiDAR channels in the dataset.
        n_class (int): The number of classes in the dataset.
        width (int): The width of the dataset's images or data grid.
        height (int): The height of the dataset's images or data grid.
        label_dict (dict[int, str]): A dictionary mapping class indices to class names.
        wavelength (np.ndarray): An array containing the wavelengths of the hyperspectral bands.
    """
    name: str
    full_name: str
    homepage: str
    n_channel_hsi: int
    n_channel_dsm: int
    width: int
    height: int
    n_class: int
    label_name: dict[int, str]
    palette: list[str]
    wavelength: np.ndarray
    version: NotRequired[str]
    license: NotRequired[str]
    # n_sample: int
    # n_train_sample: Optional[int]
    # n_val_sample: Optional[int]

__all__ = [
    'DataMetaInfo',
]