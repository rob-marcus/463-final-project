"""Class for exporting images into jpgs and pngs.
"""
import constants
from random import choice
from string import ascii_lowercase
from img import Img
from file_extensions import FileExtensions
from skimage.io import imsave
import os


class Exporter:
    @staticmethod
    def __rand_id():
        return ''.join(choice(ascii_lowercase)
                       for i in range(constants.RAND_ID_LEN))

    @classmethod
    def __build_path(cls,
                     img_obj: Img,
                     ext: FileExtensions,
                     alt_name: str = None,
                     alt_dest: str = None):
        prefix = alt_name if alt_name else img_obj.filename
        dest = alt_dest if alt_dest else img_obj.dirname

        return dest + "/" + "_".join([prefix, cls.__rand_id()]) + \
            '.' + ext.value

    @classmethod
    def export_img(cls,
                   img_obj: Img,
                   ext: FileExtensions,
                   alt_name: str = None,
                   alt_dest: str = None,
                   default_quality = constants.DEFAULT_QUALITY,
                   downsampling_rate = 1):
        path = \
            cls.__build_path(img_obj, ext, alt_name=alt_name, alt_dest=alt_dest)

        if not os.path.isdir(alt_dest):
            print("\tMaking directory {}".format(alt_dest))
            os.mkdir(path)

        print("Saving to " + path)
        if ext == FileExtensions.JPG or ext == FileExtensions.JPEG:
            imsave(path, img_obj.img[::downsampling_rate], default_quality=default_quality)
        else:
            imsave(path, img_obj.downscale_img(downsampling_rate))

    @classmethod
    def export_imgs(cls, img_objs,
                    exts,
                    alt_names=None,
                    alt_dests=None,
                    downsampling_rate=1):
        # Throw an error if mismatched lengths...
        min_length = len(img_objs)

        if len(exts) != min_length:
            print("Must have as many extensions as images.")
            print("Got {}, expected {}".format(len(exts), min_length))
            assert False

        if alt_names and len(alt_names) != min_length:
            print("Must have as many alt_names as images.")
            print("Got {}, expected {}".format(len(alt_names), min_length))

        if alt_dests and len(alt_dests) != min_length:
            print("Must have as many alt_dests as images.")
            print("Got {}, expected {}".format(len(alt_dests), min_length))

        for index, img_obj in enumerate(img_objs):
            ext = exts[index]
            alt_name = alt_names[index] if alt_names else None
            alt_dest = alt_dests[index] if alt_names else None

            cls.export_img(img_obj,
                           ext,
                           alt_name=alt_name,
                           alt_dest=alt_dest,
                           downsampling_rate=downsampling_rate)

