"""Implements a lightweight depth map class.

Note plotting the depth map requires the open3d library.
"""
from dataclasses import dataclass
import numpy as np
from typing import Tuple
from img import Img
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d


@dataclass
class DepthMap:
    d_map: np.ndarray  # H x W depth map
    c_map: Img  # H x W x 3 color map

    shape: Tuple[int, int] = None

    def __post_init__(self):
        if self.d_map.shape != self.c_map.shape[:2]:
            print("d_map and c_map must be same shape.")
            print("\td_map.shape={}".format(self.d_map.shape))
            print("\tc_map.shape={}".format(self.c_map.shape))
            assert False

        self.shape = self.d_map.shape

        # Normalize the depth map to 0, 1
        self.d_map = (self.d_map - np.min(self.d_map))/np.ptp(self.d_map)

    def plot_surface(self):
        Z = self.d_map
        H, W = Z.shape
        x, y = np.meshgrid(np.arange(0, W), np.arange(0, H))

        # set 3D figure
        fig = plt.figure()
        ax = fig.gca(projection="3d")

        # add a light and shade to the axis for visual effect # (use the ‘-’
        # sign since our Z-axis points down)
        ls = LightSource()
        color_shade = ls.shade(-Z, plt.cm.gray)

        # display a surface # (control surface resolution using rstride and
        # cstride)

        surf = ax.plot_surface(x, y, -Z, facecolors=color_shade, rstride=4,
                               cstride=4)

        # turn off axis
        plt.axis('off')
        plt.show()

    def plot_o3d(self):
        x_pts, y_pts = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))

        x_pts_flat, y_pts_flat = x_pts.flatten(), y_pts.flatten()
        d_map_flat = self.d_map.flatten()

        point_cloud = np.vstack((x_pts_flat, y_pts_flat, d_map_flat)).T

        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector((self.c_map.img /
                                                 255.0).astype(np.float64).reshape((-1, 3)))
        pcd.points = o3d.utility.Vector3dVector(
            point_cloud.astype(np.float64)
        )
        o3d.visualization.draw_geometries([pcd])
