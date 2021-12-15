"""Basic class for plotting multiple images in a grid.
"""
from img import Img
from dataclasses import dataclass
from typing import Iterable, Callable
from numpy import ndarray
import matplotlib.pyplot as plt


@dataclass
class Plotter:
    @staticmethod
    def plot(img_objs: Iterable[Img],
             shape: (int, int),
             cmap: str = None,
             op: Callable[[ndarray], ndarray] = None,
             downscale_factor: int = None,
             title: str = None):
        """Plots imgs with a title above each image in a grid defined by
        shape, and in the order the images are passed in.
        """
        if title is None:
            fig = plt.figure(figsize=(10, 10))
        else:
            fig = plt.figure(title, figsize=(10, 10))

        ax = []
        rows, columns = shape
        i_counter = 0
        for img_obj in img_objs:
            ax.append(fig.add_subplot(rows, columns, i_counter + 1))
            ax[-1].set_title(img_obj.filename)
            this_cmap = img_obj.cmap if cmap is None else cmap
            if downscale_factor is not None:
                img = img_obj.downscale_img(downscale_factor)
            else:
                img = img_obj.normalized_img

            if op is None:
                plt.imshow(img, cmap=this_cmap)
            else:
                plt.imshow(op(img))

            i_counter += 1
            plt.xticks([])
            plt.yticks([])


        plt.show()

    @classmethod
    def plot_diff(cls, img_obj_a: Img, img_obj_b: Img):
        """Plots img_a, img_b, and img_a - img_b.

        :param img_obj_a:
        :param img_obj_b:
        :return: None
        """

        diff = img_obj_a.diff(img_obj_a.img, img_obj_b.img)
        diff_img_obj = Img(diff,
                           "{a_fn} - {b_fn}.d".format(a_fn=img_obj_a.filename,
                                                      b_fn=img_obj_b.filename))

        cls.plot([img_obj_a, img_obj_b, diff_img_obj], (2, 2))

    @staticmethod
    def p(img, cmap="gray"):
        plt.imshow(img, cmap=cmap)

        plt.show()