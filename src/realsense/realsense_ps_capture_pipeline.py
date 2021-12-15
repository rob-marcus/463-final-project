"""Implements a procedure for collecting unstructured photometric stereo data from an intel realsense device
while also collecting aligned depth data (relative to the capture frame/plane.)
"""
import sys

from skimage.io import imsave

import pyrealsense2 as rs
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, List
import time
from realsense_ps_capture_object import RealsensePsCaptureObject
from img import Img


@dataclass
class RealsensePsCapturePipeline:
    frames_to_cap: int
    seconds_between_frames: float
    output_path: str
    fg_dist_meters: int = 1  # For building a mask (Mask object > fg_dist_meters away from sensor.)

    # Camera parameters.
    pipeline = None
    config = None
    profile = None

    depth_resolution: Tuple[int, int] = (1024, 768)
    image_resolution: Tuple[int, int] = (1920, 1080)
    framerate: int = 30  # Highly recommend NOT changing this value.

    # Output objects.
    depth_frame: np.ndarray = None
    image_frames: List[np.ndarray] = None

    # Some values we want to save when building the output object.
    f_x: float = None
    f_y: float = None
    ppx: float = None
    ppy: float = None
    f: float = None

    delete_first: bool = False  # Useful flag for running polarization capture so last capture is just to get an
    # unpolarized depth map and gives time to setup filter orientation.

    def __post_init__(self):
        """Initialize the camera profile/configuration."""
        print("Initializing camera pipeline.")
        # Create a pipeline
        pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        # different resolutions of color and depth streams
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("Use a combined depth + image sensor with this pipeline.")
            assert False

        config.enable_stream(rs.stream.depth,
                             *self.depth_resolution,
                             rs.format.z16,
                             self.framerate)

        config.enable_stream(rs.stream.color,
                             *self.image_resolution,
                             rs.format.bgr8,
                             self.framerate)  # Stream raw 10 bit images.

        self.pipeline = pipeline
        self.config = config

    def start(self):
        print("Starting camera.")
        self.profile = self.pipeline.start(self.config)
        print("Starting capture.")
        self.capture_frames()
        print("Ending capture.")

    def capture_frames(self):
        color_images: List[np.ndarray] = []  # Left as iterable.
        depth_images: List[np.ndarray] = []  # We will take the mean of this to reduce potential noise.

        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        print("Depth Scale is: ", depth_scale)

        fg_dist = self.fg_dist_meters / depth_scale

        align_to = rs.stream.color  # Align our depth to our rgb sensor.
        aligner = rs.align(align_to)  # Gets called as aligner.process to handle alignment.

        self.f_x = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics().fx
        self.f_y = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics().fy
        self.ppx = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics().ppx
        self.ppy = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics().ppy

        self.f = depth_scale

        print()
        d_img = None
        d_img_3d = None
        try:
            frame_no = 0
            while frame_no < self.frames_to_cap:
                sys.stdout.write("\033[K")  # Clear the previous printed line.
                print("Capturing a frame. Don't change lighting until this line is replaced.")

                frames = self.pipeline.wait_for_frames()

                if frame_no == 0:
                    aligned_frames = aligner.process(frames)

                    depth_frame = aligned_frames.get_depth_frame()  # Hd x Wd
                    color_frame = aligned_frames.get_color_frame()  # Hc x Wc x 3

                    # Check that each frame is valid.
                    if not depth_frame or not color_frame:
                        continue  # Reset.

                    d_img: np.ndarray = np.asanyarray(depth_frame.get_data())
                    d_img_3d = np.dstack((d_img, d_img, d_img))
                else:
                    color_frame = frames.get_color_frame()

                    if not color_frame:
                        continue

                c_img: np.ndarray = np.asanyarray(color_frame.get_data())

                # Clip pixels outside of the fg_dist.
                d_img_bg_removed = np.where(np.bitwise_or(d_img > fg_dist, d_img <= 0), 0, d_img)
                c_img_bg_removed = np.where(np.bitwise_or(d_img_3d > fg_dist, d_img_3d <= 0), 0, c_img)

                depth_images.append(d_img_bg_removed)
                color_images.append(c_img_bg_removed)

                # Sleep while we wait for the lighting to change.
                frame_no += 1
                sys.stdout.write("\033[K")  # Clear the previous printed line.
                print("Frame {} captured. Waiting {} seconds till next frame."
                      .format(frame_no, self.seconds_between_frames))
                time.sleep(self.seconds_between_frames)
            print("All frames captured")
        finally:
            self.pipeline.stop()

        # Just take depth from the last frame captured.
        self.depth_frame = (depth_images[0] * fg_dist) + 0.000001 # some eps.
        # Normalize the depth frame.
        self.depth_frame = (self.depth_frame - np.min(self.depth_frame))/np.ptp(self.depth_frame)
        if self.delete_first:
            self.image_frames = color_images[1:]
        else:
            self.image_frames = color_images

    def save_linear_images(self):
        for i, image in enumerate(self.image_frames):
            path = self.output_path + str(i).zfill(6) + ".png"
            img_obj = Img(image, path)

            # We want to save the linearized version of the img_obj as we cannot access the raw image from the realsense
            # Device.
            imsave(path, img_obj.linearized_img)
            print("Saved output to {}".format(path))

    def save_depth_frame(self):
        path = self.output_path + "depth_frame.png"
        imsave(path, self.depth_frame)
        print("Saved depth_frame to {}".format(path))

    def to_raw_object(self):
        self.save_depth_frame()
        self.save_linear_images()
        return RealsensePsCaptureObject(f_x=self.f_x,
                                        f_y=self.f_y,
                                        ppx=self.ppx,
                                        ppy=self.ppy,
                                        f=self.f,
                                        output_path=self.output_path)
