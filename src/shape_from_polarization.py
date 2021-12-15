"""Implements shape from polarization.

Some code for this section was motivated by annotated reading of the following papers:
1 - https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.418.3277&rep=rep1&type=pdf
    - Polarization-based Inverse Rendering from a Single View, Miyazaki et al.
2 - https://ieeexplore.ieee.org/document/1541272
    - Multi-view surface reconstruction using polarization, Atkinson et al
3 - https://cseweb.ucsd.edu/~ravir/pamipolarize_final.pdf
    - Height-from-Polarisation with Unknown Lighting or Albedo, Smith Et Al
    - Atkinson has a 5 point algorithm which outlines the broadstroke details needed to go from
    raw input -> phase/degree of pol -> ... -> depth map, where ... is pretty complicated steps.
    - Smith complements this with additional details on disambiguation.
4 - Linear depth estimation from an uncalibrated, monocular polarisation image.
    - William A. P. Smith, Ravi Ramamoorthi, and Silvia Tozza.
    - Primary method for implementing SfP in this code.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import curve_fit
from skimage.color import rgb2xyz

from img import Img


@dataclass
class ImgParametrization:
    # Sized shape...
    intensities: np.ndarray
    rhos: np.ndarray  # Degree of polarization
    phis: np.ndarray  # Azimuth
    shape: Tuple[int, int]

    _theta: np.ndarray = None  # Zenith angle.
    n: float = 1.5

    def __post_init__(self):
        self.intensities = np.reshape(self.intensities, self.shape)
        self.rhos = np.reshape(self.rhos, self.shape)
        self.phis = np.reshape(self.phis, self.shape)

    @property
    def theta(self):
        if self._theta is None:
            aa = (self.n - 1/self.n)**2 + self.rhos * (self.n + (1/self.n))**2
            bb = 4 * self.rhos * ((self.n**2) + 1) * (aa - 4 * self.rhos)
            cc = bb**2 + 16 * (self.rhos**2) * (16 * (self.rhos**2) - aa**2) * ((self.n**2) - 1)**2
            dd = np.sqrt(np.real((-bb - np.sqrt(cc.clip(0)))/(2 * (16 * (self.rhos**2) - aa**2))).clip(0))
            self._theta = np.arcsin(dd)
            """
            # We implicitly assume n=1.5 for solving the fresnel equations to recover the zenith angle theta.
            # We want to solve eq 3 of Kadambi et al 2017 for theta.
            # This is quite trivial if we put theta in a closed form, which can be done by mathematica for example
            # rather easily for a fixed n.
            inner_sqrt_num = 1.90493 * self.rhos ** 2 - 1.90493 * self.rhos ** 4
            inner_sqrt_den = (0.0798722 + 1.07987 * self.rhos + self.rhos ** 2) ** 2
            inner_sqrt = np.sqrt((inner_sqrt_num / inner_sqrt_den).clip(0))

            outer_sqrt_left_hand_num = 0.319489 + 1.32907 * self.rhos + 1.00958 * self.rhos ** 2
            outer_sqrt_left_hand_den = 0.0798722 + 1.07987 * self.rhos + self.rhos ** 2
            outer_sqrt_left_hand_term = outer_sqrt_left_hand_num / outer_sqrt_left_hand_den

            outer_sqrt = np.sqrt((outer_sqrt_left_hand_term - 2 * inner_sqrt).clip(0))

            out = -1 * np.arccos(np.clip(-0.5 * outer_sqrt, -1., 1.))

            self._theta = out"""

        return self._theta


@dataclass
class ShapeFromPolarization:
    input_images: List[Img]  # Input list of n linearized input images.
    rotations: np.ndarray  # should be length n

    image_mat: np.ndarray = None

    _img_parametrization: ImgParametrization = None
    _normals: np.ndarray = None

    def __post_init__(self):
        if len(self.input_images) == 0:
            print("Need a non-zero number of images to run.")
            assert False

        self.shape = (self.input_images[0].shape[0], self.input_images[0].shape[1])
        self.mat_dim = self.shape[0] * self.shape[1]

        self.image_mat = np.vstack([rgb2xyz(img_obj.img)[:, :, 1].flatten() for img_obj in self.input_images])

    @property
    def img_parametrization(self):
        # 4.1 in Atkinson
        if self._img_parametrization is None:
            depolarized_intensities = np.zeros(self.mat_dim)
            rhos = np.zeros_like(depolarized_intensities)
            phis = np.zeros_like(depolarized_intensities)

            # Each column in image_mat corresponds to some pixel p
            # i.e., p = image_mat[:, j]
            # p[0] is the intensity corresponding to rotations[0], p[1] the intensity corresponding to rotations[1], etc.
            # We want to fit a sine wave to each pixel p.
            # and from that, we will get the params for the fitted sine wave, and we can compute our values.

            def f(x, offset, ampl, phase):
                return np.sin(x + phase) * ampl + offset

            for col_idx in range(self.image_mat.shape[1]):  # Sadly no way to curve fit by axis...
                # Motivated by https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
                # And https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
                data = self.image_mat[:, col_idx]

                guess_offset = np.mean(data)
                guess_amplitude = 3 * np.std(data + 0.00001) / (2 ** 0.5)
                guess_phase = 0

                out = curve_fit(f,
                                self.rotations,
                                data,
                                p0=np.array([guess_offset, guess_amplitude, guess_phase]),
                                )

                optimized_params = out[0]
                optimized_outs = f(self.rotations, *optimized_params)
                if col_idx % 1000 == 0:
                    print("On col_idx={}, {}% done.".format(col_idx, 100*round(col_idx/self.image_mat.shape[1], 4)))

                """
                depolarized_intensities[col_idx] = optimized_outs[0] + optimized_outs[2]
                i_max = depolarized_intensities[col_idx] + np.min(optimized_outs)
                i_min = depolarized_intensities[col_idx] - np.min(optimized_outs)

                rhos[col_idx] = (i_max - i_min) / (i_max + i_min + 0.00001)
                mean_emission = np.mean(optimized_outs)
                phis[col_idx] = mean_emission if mean_emission >= 0 else mean_emission + np.pi
                """
                i_max = np.max(optimized_outs)
                i_min = np.min(optimized_outs)

                phis[col_idx] = optimized_params[-1]
                rhos[col_idx] = (i_max + i_min) / (i_max - i_min + 0.0001)
                depolarized_intensities[col_idx] = (i_max + i_min) / 2.
            phis %= np.pi

            self._img_parametrization = ImgParametrization(depolarized_intensities, rhos, phis, self.shape)

        return self._img_parametrization

    @property
    def normals(self):
        """Invariant lighting condition means we don't have to do any crazy propagation solving.

        """
        if self._normals is None:
            # zenith_angles = np.tan(self.img_parametrization.theta)
            zenith_angles_cos = np.cos(self.img_parametrization.theta)
            zenith_angles_sin = np.sin(self.img_parametrization.theta)

            azimuth_angles_cos = np.cos(self.img_parametrization.phis)
            azimuth_angles_sin = np.sin(self.img_parametrization.phis)

            # xs = zenith_angles * azimuth_angles_cos
            # ys = zenith_angles * azimuth_angles_sin
            # zs = np.ones_like(zenith_angles)
            xs = azimuth_angles_sin * zenith_angles_sin
            ys = azimuth_angles_cos * zenith_angles_sin
            zs = zenith_angles_cos

            return np.dstack((xs, ys, zs))

        return self._normals
