"""Implements a procedure for combining coarse depth maps and surface normals.

Sources:
1 - "Efficiently Combining Positions and Normals for Precise 3D Geometry" by Nehab Et Al.
2 - https://www-users.cs.york.ac.uk/~wsmith/papers/yu2019depth.pdf
    - Ye Yu and William A. P. Smith. Depth estimation meets inverse rendering for single image novel view synthesis.
        In Proc. CVMP, 2019.
    - They provide a more explicit formulation for building the linear system used to merge normals and depth.
"""
from dataclasses import dataclass
import numpy as np
from depth_map import DepthMap
from img import Img
from scipy.sparse import csr_matrix, identity, vstack, hstack
from scipy.sparse.linalg import lsqr


@dataclass
class MultiModalDepth:
    d_map: DepthMap
    normals: np.ndarray

    f_x: float  # Camera focal length in pixels WRT x.
    f_y: float  # Camera focal length in pixels WRT y.
    f: float  # Camera focal length.

    v_lambda: float  # Weight to place on depth map vs normals.

    _merged_depth: DepthMap = None
    _diff: np.ndarray = None
    _diff_img_obj: Img = None

    @property
    def merged_depth(self):
        if self._merged_depth is None:
            self._merged_depth = self.d_map

            # Get gradient of map in x and y direction
            dz_dx = np.gradient(self.d_map.d_map, axis=1)
            dz_dy = np.gradient(self.d_map.d_map, axis=0)

            x_pts, y_pts = np.meshgrid(np.arange(0, self.d_map.shape[1]),
                                       np.arange(0, self.d_map.shape[0]))

            mat_dim = self.d_map.shape[0] * self.d_map.shape[1]

            X = self.sparse(np.arange(mat_dim), np.arange(mat_dim),
                            x_pts.flatten(),
                            mat_dim, mat_dim)
            Y = self.sparse(np.arange(mat_dim), np.arange(mat_dim),
                            y_pts.flatten(),
                            mat_dim, mat_dim)

            dz_dx = self.sparse(np.arange(mat_dim), np.arange(mat_dim),
                                dz_dx.flatten(),
                                mat_dim, mat_dim)
            dz_dy = self.sparse(np.arange(mat_dim), np.arange(mat_dim),
                                dz_dy.flatten(),
                                mat_dim, mat_dim)

            t_x_term_1 = np.array([
                (-1. / self.f_x) * X,
                (-1. / self.f_x) * identity(mat_dim)
            ])
            t_x_term_2 = np.array([
                (-1. / self.f_y) * Y,
                self.sparse_zeros(mat_dim, mat_dim)
            ])
            t_x_term_3 = np.array([
                identity(mat_dim),
                self.sparse_zeros(mat_dim, mat_dim)
            ])

            t_x_lhs = np.array([
                t_x_term_1,
                t_x_term_2,
                t_x_term_3
            ])

            t_x_rhs = np.array([
                [dz_dx],
                [identity(mat_dim)]
            ])

            t_x = np.dot(t_x_lhs, t_x_rhs)

            t_y_term_1 = np.array([
                (-1. / self.f_x) * X,
                self.sparse_zeros(mat_dim, mat_dim)
            ])
            t_y_term_2 = np.array([
                (-1. / self.f_y) * Y,
                (-1. / self.f_y) * identity(mat_dim)
            ])
            t_y_term_3 = np.array([
                identity(mat_dim),
                self.sparse_zeros(mat_dim, mat_dim)
            ])

            t_y_lhs = np.array([
                t_y_term_1,
                t_y_term_2,
                t_y_term_3
            ])

            t_y_rhs = np.array([
                [dz_dy],
                [identity(mat_dim)]
            ])

            t_y = np.dot(t_y_lhs, t_y_rhs)

            formatted_normals = self.sparse(
                np.array([np.arange(mat_dim), np.arange(mat_dim), np.arange(mat_dim)]).flatten(),
                np.arange(3 * mat_dim).flatten(),
                np.array([
                    self.normals[:, :, 0].flatten(), self.normals[:, :, 1].flatten(), self.normals[:, :, 2].flatten()
                ]).flatten(),
                mat_dim,
                3 * mat_dim
            )
            # return formatted_normals, t_x, t_y

            lhs_term_1 = self.v_lambda * identity(mat_dim)
            lhs_term_2 = formatted_normals * vstack(t_x.reshape(3, ))
            lhs_term_3 = formatted_normals * vstack(t_y.reshape(3, ))

            lhs = np.array([
                [lhs_term_1],
                [lhs_term_2],
                [lhs_term_3],
            ])

            rhs_term = np.hstack((self.d_map.d_map.flatten(), np.zeros(2 * mat_dim)))
            rhs_term = rhs_term.reshape(-1,)
            lhs = vstack(lhs.reshape(3,))

            print("Solving the least squares problem. This may take a minute for small lambda values.")

            out = lsqr(lhs, rhs_term)
            z = out[0].reshape(self.d_map.shape)

            self._merged_depth = DepthMap(z, self.d_map.c_map)

        return self._merged_depth

    @property
    def diff(self):
        if self._diff is None:
            self._diff = self.merged_depth - self.d_map.d_map

        return self._diffw

    @staticmethod
    def sparse(i, j, v, m, n):
        """Equivalent to matlab's sparse(i, j, v, m, n)
        """
        x = csr_matrix((v, (i, j)), shape=(m, n), dtype=np.float32)
        return x

    @staticmethod
    def sparse_zeros(m, n):
        x = csr_matrix((m, n), dtype=np.float32)
        return x

    @property
    def diff_img_obj(self):
        if self._diff_img_obj is None:
            old_name = self.d_map.c_map.filename

            self._diff_img_obj = Img(self.diff, "improved_d_map_diff_wrt_{}.d".format(old_name))

        return self._diff_img_obj
