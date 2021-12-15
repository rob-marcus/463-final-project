from importer import Importer
from photometric_stereo import UncalibratedPhotometricStereo, \
    ApproximatePhotometricStereo, CalibratedPhotometricStereo
from file_extensions import FileExtensions
import numpy as np
from img import Img
from plotter import Plotter
# from simple_renderer import SimpleRender, LightingDirection
from typing import Tuple, Union, List
from exporter import Exporter
from file_extensions import FileExtensions


def get_albedo_img(ps_instance: Union[
    UncalibratedPhotometricStereo, ApproximatePhotometricStereo,
    CalibratedPhotometricStereo]) -> Img:
    albedos_obj = Img(ps_instance.albedos.reshape(ps_instance.img_shape),
                      "{}_albedos.d".format(str(ps_instance)),
                      cmap="gray")

    return albedos_obj


def get_normals_img(ps_instance: Union[
    UncalibratedPhotometricStereo, ApproximatePhotometricStereo,
    CalibratedPhotometricStereo]) -> Img:
    # Normalize the range of the normals to [0, 1] from [-1, 1].
    # Reshape
    normals = ((ps_instance.normals + 1) / 2)

    # Reshape channel-wise to get 1 image. Otherwise it will be a 3x3 set of
    # normals...
    normals_r = normals[0, :].reshape(ps_instance.img_shape)
    normals_g = normals[1, :].reshape(ps_instance.img_shape)
    normals_b = normals[2, :].reshape(ps_instance.img_shape)

    normals = np.dstack((normals_r, normals_g, normals_b))
    normals_obj = Img(normals, "{}_normals.d".format(str(ps_instance)))

    return normals_obj


def run_ps(ps_instance: Union[
    UncalibratedPhotometricStereo, ApproximatePhotometricStereo,
    CalibratedPhotometricStereo]) -> Tuple[Img, Img]:
    """Puts the albedo and normal based on the photometric stereo instance in
    plot-ready img objects.
    :param ps_instance: An instantiated photometric stereo instance.
    :returns: A 2-tuple of (albedo_img_obj, normal_img_obj).
    """

    albedos_obj = get_albedo_img(ps_instance)
    normals_obj = get_normals_img(ps_instance)

    return albedos_obj, normals_obj


if __name__ == "__main__":
    imgs = Importer.load_imgs("realsense/pipeline_data/",
                              FileExtensions.PNG,
                              bbox=(380, 900, 580, 1150),
                              file_prefix="0")

    np.set_printoptions(precision=3)

    Q = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [-10, 5, 1]])

    uncalibrated_ps = UncalibratedPhotometricStereo(imgs)
    uncalibrated_ps.enforce_integrability = True
    uncalibrated_ps.sigma = 5
    approximate_ps = ApproximatePhotometricStereo(imgs, Q)
    approximate_ps.enforce_integrability = True
    approximate_ps.sigma = 5

    Plotter.plot([get_albedo_img(uncalibrated_ps),
                  get_normals_img(uncalibrated_ps),
                  uncalibrated_ps.depth_map_img(),
                  get_albedo_img(approximate_ps),
                  get_normals_img(approximate_ps),
                  approximate_ps.depth_map_img()],
                 (2, 3),
                 title="Albedos, normals, surface captured for captured data.")
