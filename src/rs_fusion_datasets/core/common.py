from typing import TypedDict
import numpy as np

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
    wavelength: np.ndarray


__all__ = [
    'DataMetaInfo',
]