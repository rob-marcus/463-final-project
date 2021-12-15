"""Basic class for managing data from imported image (img, fname, ext.)


Also supports getting a normalized version of the image.

If creating an Img instance using an intermediary result, set full_path to
whatever the img title should be, and postfix ".<anything>", i.e., ".d".
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from file_extensions import FileExtensions
from numpy import ndarray
from os.path import basename, dirname
from typing import Callable


@dataclass
class Img:
    img: ndarray
    full_path: str
    cmap: str = None
    _normalized_img: ndarray = None
    _linearized_img: ndarray = None

    def __post_init__(self):
        self.ext = FileExtensions.path_to_enum(self.full_path)
        self.filename = basename(self.full_path).rsplit('.', 1)[0]
        self.dirname = dirname(self.full_path)
        self.shape = self.img.shape
        self.channels = self.shape[2] if len(self.shape) == 3 else 1

    def __repr__(self):
        return self.full_path

    @staticmethod
    def __normalize_img(img: ndarray):
        normalized_img = (img - np.min(img)) / (np.max(img) - np.min(img))

        return normalized_img

    @property
    def normalized_img(self):
        if self._normalized_img is None:
            self._normalized_img = self.__normalize_img(self.img)

        return self._normalized_img

    @property
    def linearized_img(self):
        """Retrieves the normalized linearized ***normalized*** image.

        :return: normalized linearized version of normalized_img.
        """
        if self._linearized_img is None:
            img = self.normalized_img
            non_linear_bound = 0.0404482
            img_linear = np.piecewise(
                img,
                [img <= non_linear_bound,
                 img > non_linear_bound],
                [lambda nl_img: nl_img / 12.92,
                 lambda nl_img: ((nl_img + 0.055) / 1.055) ** 2.4]
            )

            self._linearized_img = self.__normalize_img(img_linear)

        return self._linearized_img

    def downscale_img(self, default_downscale_factor=20):
        return self.img[::default_downscale_factor,
                        ::default_downscale_factor]

    @staticmethod
    def diff(src_img: ndarray, alt_img: ndarray):
        return src_img - alt_img

    def plot(self,
             img: ndarray,
             cmap: str = None,
             title: str = None,
             op: Callable[[ndarray], ndarray] = None):
        if title is None:
            title = "Plot of {filename}".format(filename=self.filename)
        if cmap is None:
            cmap = self.cmap

        def apply_op():
            return img if op is None else op(img)

        plt.title(title)
        plt.imshow(apply_op(), cmap=cmap)
        plt.show()

    def __mul__(self, other):
        # TODO: ideally we would have some mechanism to update the other
        #  properties when this happens... For now NBD.
        return self.img * other

    def __add__(self, other):
        return self.img + other
