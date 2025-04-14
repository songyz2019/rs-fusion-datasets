from typing import Literal
import numpy as np

__all__ = ['hsi2rgb']

def _cie_xyz_weights(n):
    """

    n 波长
    """

    def g(x, mu, sigma1, sigma2):
        """分段高斯函数"""
        return np.where(
            x < mu,
            np.exp(-(x - mu) ** 2 / (sigma1 ** 2) / 2),
            np.exp(-(x - mu) ** 2 / (sigma2 ** 2) / 2),
        )
    x = (
        0.362   * g(n, 442.0 , 16.0 , 26.7)
        -0.065  * g(n, 501.1, 20.4 , 26.2 )
        + 1.056 * g(n, 599.8, 37.9 , 31.0 )
    )
    y = (
        0.286   * g(n, 530.9 , 16.3 , 31.1)
        + 0.821 * g(n, 568.8, 46.9 , 40.5 )
    )
    z = (
        0.980   * g(n, 437.0 , 11.8 , 36.0)
        + 0.681 * g(n, 459.0, 26.0 , 13.8 )
    )
    return x,y,z

# This should be a clear, and straightforward core functions, without any unnecessary complexity, since the algorithm itself is complex enough.
def _hsi2rgb(hsi, wavelength):
    """
    Convert an HSI (Hyperspectral Imaging) Image to RGB format. This is a raw fuctions without any checking and convertions of inputs and outputs.
    
    Parameters:
    hsi : numpy.ndarray
        The input HSI data, expected to be a 3D array of shape (height, width, spectral bands) where the last dimension represents different spectral bands.
        The values are expected to be in the range [0, 1].
    wavelength : numpy.ndarray
        The wavelengths corresponding to the spectral bands in the HSI data. Expected to be a 1D array of the same length as the number of spectral bands.
        The values are expected to be in nanometers.
    raw : bool, optional
        If True, the function will return the raw RGB values without rescaling intensity. Default is False.
    
    Returns:
    numpy.ndarray
        The RGB representation of the input HSI data, with shape (height, width, 3). The values are rescaled to the range [0, 1] if raw is False.
    """
    
    x,y,z = _cie_xyz_weights(wavelength)
    X = np.sum(hsi*x, axis=-1)
    Y = np.sum(hsi*y, axis=-1)
    Z = np.sum(hsi*z, axis=-1)

    transform_matrix = [
        [3.1338561, -1.6168667, -0.4906146],
        [-0.9787684, 1.9161415, 0.0334540],
        [0.0719453, -0.2289914, 1.4052427]
    ]

    xyz = np.stack([X, Y, Z], axis=-1)
    rgb = np.einsum('C c, ... h w c -> ... h w C', transform_matrix, xyz)

    return rgb


# This should be a wrapper for _hsi2rgb, make it user-friendly and compatible with many inputs.
def hsi2rgb(hsi, wavelength_range=(350,1000), wavelength=None, input_format: Literal['CHW', 'HWC'] = 'CHW', output_format: Literal['CHW', 'HWC', 'Same'] = 'Same', xyz_to_rgb=True, gamma=2.2):
    """
    Converts a hyperspectral image (HSI) to an RGB image. 
    Parameters:
        hsi (numpy.ndarray, torch.Tensor, or jax.numpy.ndarray): 
            The input hyperspectral image. It can be a numpy array, a PyTorch tensor, or a JAX array.
        wavelength_range (tuple, optional): 
            The range of wavelengths (in nanometers) corresponding to the hyperspectral bands. 
            Defaults to (350, 1000) which looks fine for most sensors.
        wavelength (numpy.ndarray, optional): 
            In most case, you just need wavelength_range, and leave wavelength None. A 1D array specifying the exact wavelengths for each band. If None, it will be 
            generated as a linear space within `wavelength_range`. Defaults to None.
        input_format (Literal['CHW', 'HWC'], optional): 
            The format of the input HSI. 'CHW' indicates (Channels, Height, Width), and 'HWC' 
            indicates (Height, Width, Channels). Defaults to 'CHW'.
        output_format (Literal['CHW', 'HWC', 'Same'], optional): 
            The desired format of the output RGB image. 'Same' will use the same format as 
            `input_format`. Defaults to 'Same'.
        xyz_to_rgb (bool, optional): 
            Whether to normalize and apply gamma correction to the RGB output. Defaults to True.
        gamma (float, optional): 
            The gamma correction factor to apply if `xyz_to_rgb` is True. Defaults to 2.2.
    Returns:
        numpy.ndarray, torch.Tensor, or jax.numpy.ndarray: 
            The converted RGB image in the specified `output_format`. The type of the output 
            matches the type of the input HSI.
    Raises:
        ValueError: 
            If the input HSI is not a numpy array, PyTorch tensor, or JAX array.
        ValueError: 
            If `input_format` or `output_format` is not one of the allowed values.
    Notes:
        - The function automatically normalizes the input HSI to the range [0, 1] if its values 
          are outside this range.
        - The RGB output is always normalized to the range [0, 1] before gamma correction.
        - The function supports batch dimensions, which are preserved in the output.
        - These doc is AI generated and audited by human
    """
    if output_format == 'Same':
        output_format = input_format

    tensor_type = 'numpy'
    if not isinstance(hsi, np.ndarray):
        # Guess the type of hsi and convert to numpy array
        try:
            import torch
            # Note: This is a little trick beacause we don't add torch or jax as a dependency, it will make this package very large.
            if isinstance(hsi, torch.Tensor):
                hsi = hsi.cpu().detach().numpy()
                tensor_type = 'torch'
        except ImportError:
            pass
    if not isinstance(hsi, np.ndarray):    
        try:
            import jax
            if isinstance(hsi, jax.numpy.ndarray):
                hsi = np.array(hsi)
                tensor_type = 'jax'
        except ImportError:
            pass
    if not isinstance(hsi, np.ndarray): 
        raise ValueError("hsi should be a numpy array, a torch tensor or a jax array")

    b=range(hsi.ndim - 3) # the batch dimension**s**
    # Convert any input shape to HWC
    if input_format == 'CHW':
        hsi = np.transpose(hsi, (*b, -2, -1, -3))
    elif input_format == 'HWC':
        pass
    else:
        raise ValueError("format should be 'HWC' or 'CHW'")
    
    if hsi.max() > 1 or hsi.min() < 0:
        max_hsi = np.max(hsi, axis=(-2, -3), keepdims=True)
        min_hsi = np.min(hsi, axis=(-2, -3), keepdims=True)
        hsi = (hsi - min_hsi)/(max_hsi-min_hsi)
    # --------- HWC Begin -----------
    if wavelength is None:
        wavelength = np.linspace(wavelength_range[0], wavelength_range[-1], hsi.shape[-1])
        # Use wavelength_range[-1] instead of wavelength_range[1] in case users pass the wrong parameter.
    rgb = _hsi2rgb(hsi, wavelength)
    if xyz_to_rgb:
        max_rgb = np.max(rgb, axis=(-2, -3), keepdims=True)
        min_rgb = np.min(rgb, axis=(-2, -3), keepdims=True)
        rgb = (rgb - min_rgb)/(max_rgb-min_rgb)
        rgb = np.power(rgb/float(np.max(rgb)), 1/gamma) if gamma != 1 else rgb
    # --------- HWC End -----------

    # Note that rgb is always hwc here
    if output_format=='CHW':
        rgb = np.transpose(rgb, (*b, -1, -3, -2))

    if tensor_type == 'torch':
        import torch
        rgb = torch.from_numpy(rgb)
    elif tensor_type == 'jax':
        import jax
        rgb = jax.device_put(rgb)
    return rgb
