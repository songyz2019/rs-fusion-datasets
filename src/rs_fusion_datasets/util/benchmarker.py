from typing import List, Optional
import warnings
import numpy as np
import skimage
from torch import Tensor, zeros
from torch import argmax, zeros_like
import torch
from .lbl2rgb import lbl2rgb
from jaxtyping import UInt8, Float
from scipy.sparse import spmatrix


class Benchmarker:
    def __init__(self, truth: spmatrix, n_class: Optional[int] = None, dataset_name: Optional[str] = None, device='cpu', dtype=torch.int):
        if n_class is None:
            self.n_class = truth.max().item()
        else:
            self.n_class = n_class
        self.device = device  # TODO: add device support
        self.dtype = dtype  # TODO: add device support
        self.TRUTH = torch.from_numpy(truth.todense()).to(device, self.dtype) # 0==background real_labels starts from 1
        self.TRUTH[self.TRUTH == -1] = 0 # Well, wierd bug: sometimes truth.todense() returns -1 and sometimes 0
        self.dataset_name = dataset_name
        self.confusion_matrix_enabled = True  # row is truth, column is prediction
        self.reset()

    def reset(self):
        self.predict = zeros_like(self.TRUTH, dtype=self.dtype, device=self.device)  # 0==background real_labels starts from 1
        self.confusion_matrix = zeros((self.n_class, self.n_class), dtype=torch.long, device=self.device)

    def predicted_image(self) -> UInt8[Tensor, "C H W"]:
        with torch.no_grad():
            img = lbl2rgb(self.predict, self.dataset_name)
            img = (img * 255.0).astype(np.uint8)
            return img

    def conf(self, normalize=False) -> UInt8[Tensor, "C C"]:
        with torch.no_grad():
            if not self.confusion_matrix_enabled:
                raise ValueError("Can not compute confusion matrix. You should **ALWAYS** add lbl_target argument when calling add_sample to enable it.")
            if normalize:
                return self.confusion_matrix.float() / self.confusion_matrix.sum(dim=-1, keepdim=True)
            else:
                return self.confusion_matrix

    def ca(self) -> Float[Tensor, "C"]:
        with torch.no_grad():
            if not self.confusion_matrix_enabled:
                raise ValueError("Can not compute confusion matrix. You should **ALWAYS** add lbl_target argument when calling add_sample to enable it.")
            return self.confusion_matrix.diag() / self.confusion_matrix.sum(dim=-1)

    def aa(self) -> Tensor:
        with torch.no_grad():
            if not self.confusion_matrix_enabled:
                raise ValueError("Can not compute confusion matrix. You should **ALWAYS** add lbl_target argument when calling add_sample to enable it.")
            return self.ca().mean()

    def oa(self) -> Tensor:
        with torch.no_grad():
            if not self.confusion_matrix_enabled:
                raise ValueError("Can not compute confusion matrix. You should **ALWAYS** add lbl_target argument when calling add_sample to enable it.")
            return self.confusion_matrix.trace() / self.confusion_matrix.sum()

    def kappa(self) -> Tensor:
        with torch.no_grad():
            if not self.confusion_matrix_enabled:
                raise ValueError("Can not compute confusion matrix. You should **ALWAYS** add lbl_target argument when calling add_sample to enable it.")
            # TODO: improve the performance of kappa
            s = torch.sum(
                torch.sum(self.confusion_matrix, dim=-1) * torch.sum(self.confusion_matrix, dim=-2)
            )
            pe = s / (torch.sum(self.confusion_matrix) ** 2)
            kappa = (self.oa() - pe) / (1 - pe)
            return kappa

    def add_sample(self, location: List[Float[Tensor, 'B I']], lbl_input: Float[Tensor, 'B D'], lbl_target: Optional[Float[Tensor, 'B D']] = None):
        with torch.no_grad():
            # preprocess arguments
            lbl_input = argmax(lbl_input, dim=-1).to(self.device, self.confusion_matrix.dtype)  # start with 0
            if lbl_target is None:
                self.confusion_matrix_enabled = False
            # add to prediction image
            x, y = location[0].to(self.device, dtype=torch.int), location[1].to(self.device, dtype=torch.int)
            self.predict[x, y] = lbl_input.to(self.device, self.dtype) + 1
            # add to confusion matrix
            if self.confusion_matrix_enabled:
                lbl_target = argmax(lbl_target, dim=-1).to(self.device, self.confusion_matrix.dtype)
                indices = self.n_class * lbl_target + lbl_input
                cm = torch.bincount(indices, minlength=self.n_class ** 2)
                cm = cm.reshape(self.n_class, self.n_class)
                self.confusion_matrix += cm

    def __call__(self, location: List[Float[Tensor, 'B I']], lbl_input: Float[Tensor, 'B D'], lbl_target: Optional[Float[Tensor, 'B D']] = None):
        self.add_sample(location, lbl_input, lbl_target)

    def frac(self) -> dict:
        """Not recommended for using"""
        with torch.no_grad():
            correct = self.confusion_matrix.diag()
            total = self.confusion_matrix.sum(dim=-1)
            return {k: f"{c}/{t}" for k, c, t in zip(range(self.n_class), correct, total)}

    def error_image(self, underlying: Optional[UInt8[np.ndarray, 'C H W']] = None) -> UInt8[np.ndarray, 'C H W']:
        """
        Not recommended for using
        """
        with torch.no_grad():
            error = self.predict != self.TRUTH
            correct = torch.logical_and(self.predict == self.TRUTH, self.TRUTH != 0)
            img = correct.to(self.device ,torch.int16) + 2 * error.to(self.device ,torch.int16)
            img = img.cpu().numpy()
            img = skimage.color.label2rgb(img, colors=['green', 'red'], alpha=0.5, bg_label=0, image=underlying, channel_axis=-3)
            img = (img * 255.0).astype(np.uint8)
            return img

    def predict_image(self, *args, **kwargs):
        warnings.warn("predict_image is deprecated, please use predicted_image instead.", DeprecationWarning)
        return self.predicted_image(*args, **kwargs)

    def confusion_plot(self):
        """
        Not recommended for using
        Return a matplotlib figure, we can't add type notions because we use matplotlib as a soft dependency
        This method depends on skimage and matplotlib softly, you need to install them manually
        """
        import matplotlib.pyplot as pl
        from sklearn.metrics import ConfusionMatrixDisplay
        fig, ax = pl.subplots(figsize=(20, 20))
        confusion = self.confusion_matrix
        disp = ConfusionMatrixDisplay(confusion.int().numpy())
        disp.plot(ax=ax)
        r = pl.gcf()
        pl.close(fig)
        return r

    def aio4paper(self):
        return f"OA: {self.oa().round(decimals=4) * 100:.02f}% \nAA: {self.aa().round(decimals=4) * 100:.02f}% \nKappa: {self.kappa().round(decimals=4) * 100:.02f}% \nCA: {' '.join(['%.02f%%' % x for x in self.ca().round(decimals=4) * 100])}"
