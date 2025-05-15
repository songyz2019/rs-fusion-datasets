import numpy as np
from torch import Tensor
from torch import argmax
from .lbl2rgb import lbl2rgb


class ClassificationMapper:
    def __init__(self, truth, dataset_name=None):
        self.dataset_name = dataset_name
        self.TRUTH = truth.todense()
        self.predict = np.zeros_like(self.TRUTH)

    def error(self):
        raise NotImplementedError("not implemented yet")
        return self.predict != self.TRUTH

    def correct(self):
        raise NotImplementedError("not implemented yet")
        return np.logical_and(self.predict == self.TRUTH, self.TRUTH != 0)

    def predict_image(self, format='chw'):
        img = lbl2rgb(self.predict, self.dataset_name)
        img = (img*255.0).astype(np.uint8)
        if format == 'hwc':
            return img.transpose(1, 2, 0)
        elif format == 'chw':
            return img
        else:
            raise ValueError(f"Unknown format: {format}")

    def add_sample(self, location, class_id, de_onehot=True):
        if de_onehot and isinstance(class_id, Tensor) and len(class_id.shape) > 1:
            class_id = argmax(class_id, dim=-1)
        self.predict[location[0].int().cpu(), location[1].int().cpu()] = class_id.cpu() + 1

    def __call__(self, location, class_id, de_onehot=True):
        self.add_sample(location, class_id, de_onehot)

    # def overlay_correct_image(self, underlying=None):
    #     return skimage.color.label2rgb(self.correct_image() + 2*self.error_image(), colors=['green', 'red'], alpha=0.5, bg_label=0, image=underlying)
