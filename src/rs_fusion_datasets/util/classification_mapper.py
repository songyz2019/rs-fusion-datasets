from typing import List, Optional
import numpy as np
from torch import Tensor, zeros
from torch import argmax, zeros_like
import torch
from .lbl2rgb import lbl2rgb
from jaxtyping import UInt8, Float
from scipy.sparse import spmatrix


class ClassificationMapper:
    def __init__(self, truth: spmatrix, n_class:Optional[int]=None, dataset_name :Optional[str]=None, device='cpu'):
        if n_class is None:
            self.n_class = truth.max().item()
        else:
            self.n_class = n_class
        self.TRUTH = torch.from_numpy(truth.todense()).to(device)
        self.dataset_name = dataset_name
        self.device = device # TODO: add device support
        self.reset()

    def reset(self):
        self.predict = zeros_like(self.TRUTH, dtype=torch.int, device=self.device)
        self.confusion_matrix = zeros((self.n_class, self.n_class), dtype=torch.long, device=self.device)

    def predict_image(self, format='chw') -> UInt8[Tensor, "C H W"]:
        img = lbl2rgb(self.predict, self.dataset_name)
        img = (img*255.0).astype(np.uint8)
        if format == 'hwc':
            return img.transpose(1, 2, 0)
        elif format == 'chw':
            return img
        else:
            raise ValueError(f"Unknown format: {format}")

    def ca(self) -> Float[Tensor, "C"]:
        return self.confusion_matrix.trace() / self.confusion_matrix.sum(dim=-1)
    
    def aa(self) -> Tensor:
        return self.ca().mean()
    
    def oa(self) -> Tensor:
        return self.confusion_matrix.trace().sum() / self.confusion_matrix.sum()
    
    def kappa(self) -> Tensor:
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
            lbl_target = zeros_like(lbl_input, device=self.device, dtype=self.confusion_matrix.dtype)  # mock a dummy target
        lbl_target = argmax(lbl_target, dim=-1).to(self.device, self.confusion_matrix.dtype)           # start with 0
        # add to prediction image
        x, y = location[0].to(self.device, dtype=torch.int), location[1].to(self.device, dtype=torch.int)
        self.predict[x, y] = lbl_input.to(self.device, self.predict.dtype) + 1
        # add to confusion matrix
        self.confusion_matrix[lbl_target, lbl_input] += 1


    def __call__(self, location, class_id):
        self.add_sample(location, class_id)

    def overlay_correct_image(self, underlying=None):
        """Not recommended for using"""
        import skimage
        error = self.predict != self.TRUTH
        correct = torch.logical_and(self.predict == self.TRUTH, self.TRUTH != 0)
        return skimage.color.label2rgb(correct + 2*error, colors=['green', 'red'], alpha=0.5, bg_label=0, image=underlying)
    
