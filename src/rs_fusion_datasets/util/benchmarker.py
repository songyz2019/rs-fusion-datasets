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
    def __init__(self, truth: spmatrix, n_class:Optional[int]=None, dataset_name :Optional[str]=None, device='cpu'):
        if n_class is None:
            self.n_class = truth.max().item()
        else:
            self.n_class = n_class
        self.TRUTH = torch.from_numpy(truth.todense()).to(device, torch.int)
        self.dataset_name = dataset_name
        self.device = device # TODO: add device support
        self.confusion_matrix_enabled = True
        self.reset()

    def reset(self):
        self.predict = zeros_like(self.TRUTH, dtype=torch.int, device=self.device)
        self.confusion_matrix = zeros((self.n_class, self.n_class), dtype=torch.long, device=self.device)

    def predicted_image(self, format='chw') -> UInt8[Tensor, "C H W"]:
        img = lbl2rgb(self.predict, self.dataset_name)
        img = (img*255.0).astype(np.uint8)
        if format == 'hwc':
            return img.transpose(1, 2, 0)
        elif format == 'chw':
            return img
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def conf(self, normalize=False) -> UInt8[Tensor, "C C"]:
        if not self.confusion_matrix_enabled:
            raise ValueError("Can not compute confusion matrix. You should **ALWAYS** add lbl_target argument when calling add_sample to enable it.")
        if normalize:
            return self.confusion_matrix.float() / self.confusion_matrix.sum(dim=-1, keepdim=True)
        else:
            return self.confusion_matrix

    def ca(self) -> Float[Tensor, "C"]:
        if not self.confusion_matrix_enabled:
            raise ValueError("Can not compute confusion matrix. You should **ALWAYS** add lbl_target argument when calling add_sample to enable it.")
        return self.confusion_matrix.diag() / self.confusion_matrix.sum(dim=-1)
    
    def aa(self) -> Tensor:
        if not self.confusion_matrix_enabled:
            raise ValueError("Can not compute confusion matrix. You should **ALWAYS** add lbl_target argument when calling add_sample to enable it.")
        return self.ca().mean()
    
    def oa(self) -> Tensor:
        if not self.confusion_matrix_enabled:
            raise ValueError("Can not compute confusion matrix. You should **ALWAYS** add lbl_target argument when calling add_sample to enable it.")
        return self.confusion_matrix.trace() / self.confusion_matrix.sum()
    
    def kappa(self) -> Tensor:
        if not self.confusion_matrix_enabled:
            raise ValueError("Can not compute confusion matrix. You should **ALWAYS** add lbl_target argument when calling add_sample to enable it.")
        """TODO: check the AI generated code"""
        total = self.confusion_matrix.sum()
        total_correct = self.confusion_matrix.trace().sum()
        total_row = self.confusion_matrix.sum(dim=-1)
        total_col = self.confusion_matrix.sum(dim=0)
        total_row_col = (total_row * total_col) / total
        return (total * total_correct - total_row_col.sum()) / (total**2 - total_row_col.sum())

    def add_sample(self, location :List[Float[Tensor, 'B I']], lbl_input: Float[Tensor, 'B D'], lbl_target: Optional[Float[Tensor, 'B D']]=None):
        # preprocess arguments
        lbl_input  = argmax(lbl_input, dim=-1).to(self.device, self.confusion_matrix.dtype)            # start with 0
        if lbl_target is None:
            self.confusion_matrix_enabled = False
        # add to prediction image
        x, y = location[0].to(self.device, dtype=torch.int), location[1].to(self.device, dtype=torch.int)
        self.predict[x, y] = lbl_input.to(self.device, self.predict.dtype) + 1
        # add to confusion matrix
        if self.confusion_matrix_enabled:
            lbl_target = argmax(lbl_target, dim=-1).to(self.device, self.confusion_matrix.dtype)
            self.confusion_matrix[lbl_target, lbl_input] += 1


    def __call__(self, location, class_id):
        self.add_sample(location, class_id)


    def frac(self) -> dict:
        """Not recommended for using"""
        correct = self.confusion_matrix.diag()
        total = self.confusion_matrix.sum(dim=-1)
        return {k:f"{c}/{t}" for k,c,t in zip(range(self.n_class), correct, total)}

    def error_image(self, underlying :Optional[UInt8[np.ndarray, 'C H W']]=None):
        """
        Not recommended for using
        """
        underlying = underlying.transpose(1, 2, 0)
        error = self.predict != self.TRUTH
        correct = torch.logical_and(self.predict == self.TRUTH, self.TRUTH != 0)
        img = correct + 2*error
        img = img.cpu().numpy()
        return skimage.color.label2rgb(img, colors=['green', 'red'], alpha=0.5, bg_label=0, image=underlying)
    
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
        fig, ax = pl.subplots(figsize=(20,20))
        confusion = self.conf.compute()
        disp = ConfusionMatrixDisplay(confusion.int().numpy())
        disp.plot(ax=ax)
        r = pl.gcf()
        pl.close(fig)
        return r
    
    def aio4paper(self):
        return f"OA: {self.oa().round(decimals=4)*100:.02f}% \nAA: {self.aa().round(decimals=4)*100:.02f}% \nKappa: {self.kappa().round(decimals=4)*100:.02f}% \nCA: {' '.join(['%.02f%%' % x for x in self.ca().round(decimals=4)*100])}"
