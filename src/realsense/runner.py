from realsense_ps_capture_pipeline import RealsensePsCapturePipeline
from realsense_ps_capture_object import RealsensePsCaptureObject
from plotter import Plotter


"""# For photometric stereo capture.
path = "photometric_stereo_roller_data/"
pipeline = RealsensePsCapturePipeline(frames_to_cap=25,
                                      seconds_between_frames=.5,
                                      output_path=path)
pipeline.start()
obj = pipeline.to_raw_object()
obj.write_capture_object()
"""

# For polarized data capture.
path = "polarized_roller_data/"
pipeline = RealsensePsCapturePipeline(frames_to_cap=4 + 1, # First frame aren't for the photos.
                                      seconds_between_frames=6,
                                      output_path=path,
                                      delete_first=True)

pipeline.start()
obj = pipeline.to_raw_object()
obj.write_capture_object()
