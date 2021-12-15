"""Output object from running RealsensePsCapturePipeline.start()
"""

from dataclasses import dataclass
from typing import List

import numpy as np

import imageio
import glob
from img import Img


@dataclass
class PointCloud:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


@dataclass
class RealsensePsCaptureObject:
    f_x: float  # Focal length of the image plane, as a multiple of pixel width
    f_y: float  # Focal length of the image plane, as a multiple of pixel height
    f:   float  # Depth scaler

    ppx: float
    ppy: float

    output_path: str  # Should be / terminated, i.e, "../data/raw_run1/"

    _depth_frame: np.ndarray = None
    _linear_pngs: List[Img] = None
    _angles: np.ndarray = None

    @property
    def angles(self):
        if self._angles is None:
            self._angles = np.arange(len(self.linear_pngs)) * 45
        return self._angles
    @property
    def linear_pngs(self):
        if self._linear_pngs is None:
            self._linear_pngs = self.load_tiffs()

        return self._linear_pngs

    @property
    def depth_frame(self):
        if self._depth_frame is None:
            self._depth_frame = self.load_depth_frame()

        return self._depth_frame

    def load_tiffs(self) -> List[Img]:
        png_paths = glob.glob("realsense/" + self.output_path + "0*.png")
        if len(png_paths) == 0:
            print("No pngs found at {}".format(self.output_path + "*.png"))
            assert False

        print("Loading linear_pngs from {}".format(self.output_path))
        return [Img(imageio.imread(path), path) for path in png_paths]

    def load_depth_frame(self) -> np.ndarray:
        paths = glob.glob(self.output_path + "depth_frame.png")
        if len(paths) == 0:
            print("No depth_frame found at {}".format(self.output_path + "depth_frame.png"))
            assert False

        print("Loading depth frame from {}".format(paths[0]))
        return imageio.imread(paths[0])

    def write_capture_object(self):
        path = self.output_path + "capture_object.npz"
        np.savez(path, f_x=self.f_x, f_y=self.f_y, f=self.f, ppx=self.ppx, ppy=self.ppy, output_path=self.output_path)

    @staticmethod
    def load_capture_object(npz_obj):
        if 'output_path' not in npz_obj:
            print("output_path missing.")
            assert False
        if 'f_x' not in npz_obj:
            print("f_x missing.")
            assert False
        if 'f_y' not in npz_obj:
            print("f_y missing.")
            assert False
        if 'f' not in npz_obj:
            print("f missing.")
            assert False

        return RealsensePsCaptureObject(f_x=npz_obj['f_x'][()],
                                        f_y=npz_obj['f_y'][()],
                                        f=npz_obj['f'][()],
                                        ppx=npz_obj['ppx'][()],
                                        ppy=npz_obj['ppy'][()],
                                        output_path=npz_obj['output_path'][()])

    def to_pointcloud(self):
        # Motivated by some realsense stdlib stuff.
        h, w = self.depth_frame.shape

        nx = np.linspace(0, w - 1, w)
        ny = np.linspace(0, h - 1, h)
        u, v = np.meshgrid(nx, ny)
        x = (u.flatten() - self.ppx)/self.f_x
        y = (v.flatten() - self.ppy)/self.f_y

        z = self.depth_frame.flatten() / 1000
        x *= z
        y *= z

        x = x[np.nonzero(z)]
        y = y[np.nonzero(z)]
        z = z[np.nonzero(z)]

        return PointCloud(x, y, z)
