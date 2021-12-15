from get_normals_runner_ps import get_normals_img
from file_extensions import FileExtensions
from importer import Importer
from photometric_stereo import UncalibratedPhotometricStereo, \
    ApproximatePhotometricStereo
import numpy as np
from plotter import Plotter
from depth_map import DepthMap
from realsense.realsense_ps_capture_object import RealsensePsCaptureObject
from multi_modal_depth import MultiModalDepth
from cv2 import GaussianBlur, BORDER_CONSTANT


# Load the images.
conch_path = "realsense/pipeline_data/"
conch_bbox = (355, 925, 580, 1150)
conch_lambda = 0.005

roller_path = "realsense/photometric_stereo_roller_data/"
roller_bbox = (0, 925, 790, 1120)
roller_lambda = 0.005

path_to_use = conch_path
bbox_to_use = conch_bbox
lambda_to_use = conch_lambda

imgs = Importer.load_imgs(path_to_use,
                          FileExtensions.PNG,
                          bbox=bbox_to_use,
                          file_prefix="0")
d_map = Importer.load_imgs(path_to_use,
                           FileExtensions.PNG,
                           bbox=bbox_to_use,
                           file_prefix="depth_frame")

# Load the camera intrinsics
capture_obj = RealsensePsCaptureObject.load_capture_object(np.load(path_to_use + "capture_object.npz"))

if len(d_map) != 1:
    print("More than one depth_map to choose from, please check the {} directory.".format(path_to_use))
    assert False

if len(imgs) == 0:
    print("imgs is length 0. Check the {} directory to resolve.".format(path_to_use))

# Get the normals.
Q = np.array([[1, 0, 0],
              [0, 1, 0],
              [-10, 5, 1]])

uncalibrated_ps = UncalibratedPhotometricStereo(imgs)
uncalibrated_ps.enforce_integrability = True
uncalibrated_ps.sigma = 5

normals = get_normals_img(uncalibrated_ps)

# Get the depth map object
depth_map = DepthMap(d_map[0].img, imgs[0])

mmd = MultiModalDepth(
    d_map=depth_map,
    normals=normals.img,
    f_x=capture_obj.f_x,
    f_y=capture_obj.f_y,
    f=capture_obj.f,
    v_lambda=lambda_to_use,
)

new_d_map = mmd.merged_depth
new_d_map.d_map = GaussianBlur(new_d_map.d_map,
                               (0, 0),
                               sigmaX=1,
                               sigmaY=1,
                               borderType=BORDER_CONSTANT)
Plotter.p(new_d_map.d_map - depth_map.d_map, cmap="gray")
Plotter.p(np.hstack((new_d_map.d_map, depth_map.d_map)), cmap="jet")