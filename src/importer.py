"""Class for importing images into Img objects.
"""

from skimage.io import imread
import glob
from img import Img
from file_extensions import FileExtensions
import numpy as np
import cv2
from typing import Tuple

class Importer:
    @classmethod
    def get_paths(cls,
                  folder: str,
                  extension: FileExtensions,
                  file_prefix: str = '',
                  sort_paths: bool = False):
        """Returns a list of paths in folder with matching file_prefix and
        extension.

        :param folder: Directory to get paths from.
        :param extension: Which file types to retrieve.
        :param file_prefix: Optional parameter, specify if only want files
            with a given prefix.
        :param sort_paths: Optional parameter, specify if files should be
            read in sorted or not. Files will be sorted in ascending order.
        :return: List of paths relative to the src directory.
        """
        out = glob.glob(folder + file_prefix + '*.' + extension.value)
        if sort_paths:
            out.sort()
        return out

    @classmethod
    def load_img(cls, path, bbox=None, downscale_factor=None):
        img = imread(path)
        if not bbox is None:
            img = img[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        out = Img(img, path)
        if not downscale_factor is None:
            out = Img(out.downscale_img(downscale_factor), path)
        return out

    @classmethod
    def load_imgs(cls,
                  folder: str,
                  extension: FileExtensions,
                  file_prefix: str = '',
                  sort_paths: bool = False,
                  bbox: Tuple[int, int, int, int] = None,
                  downscale_factor: int = None):
        """Returns a list containing retrieved images at specified folder,
        with specified extension and prefix, as Img objects.

        :param folder: Directory to get paths from.
        :param extension: Which file types to retrieve.
        :param file_prefix: Optional parameter, specify if only want files
            with a given prefix.
        :param sort_paths: Optional parameter, specify if files should be
            read in sorted or not. Files will be sorted in ascending order.
        :return: List of paths relative to the src directory.
        """
        out = []
        for path in cls.get_paths(folder, extension, file_prefix, sort_paths):
            out.append(cls.load_img(path, bbox, downscale_factor))

        return out

    @classmethod
    def load_video(cls, path):
        """Loads a video into an array of frames.

        Source code taken from 16-385 HW2 from spring 2021.

        :param path:
        :return:
        """
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        cap = cv2.VideoCapture(path)

        # Check if camera opened successfully
        if cap.isOpened() == False:
            print("Error opening video stream or file")

        i = 0
        # Read until video is completed
        while cap.isOpened():
            print(i)
            # Capture frame-by-frame
            i += 1
            ret, frame = cap.read()
            if ret == True:

                # Store the resulting frame
                if i == 1:
                    frames = frame[np.newaxis, ...]
                else:
                    frame = frame[np.newaxis, ...]
                    frames = np.vstack([frames, frame])
                    frames = np.squeeze(frames)

            else:
                break

        # When everything done, release the video capture object
        cap.release()

        return frames
