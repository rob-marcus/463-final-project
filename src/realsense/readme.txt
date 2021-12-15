This project uses intel realsense to collect data. Intel realsense is a massive PITA.

I recommend downloading pyrealsense2. Note, pyrealsense2 is available for macOS machines, but you must manually build
the library from source. I do not recommend doing this, unless you wish to burn 4-6 hours and give up anyway.

Instead, you should find an old Windows PC, and just install it through pip or conda. I have isolated the realsense
library code from the rest of the source code so individuals can collect data on a Windows machine, save the serialized
intermediate data, and then resume work from the machine of their choice.

Note, if you have never done windows development before, it is also a bit of a quagmire with environments, and python
path variables randomly disappearing.

Note, this library is currently designed to work with unstructured photometric stereo, and, as such, the runner code
is built around this procedure.

Common error:
    File "C:\Users\robbi\Desktop\15463fp\src\realsense\realsense_ps_capture_pipeline.py", line 82, in start
        self.profile = self.pipeline.start(self.config)
    RuntimeError: Couldn't resolve requests
Check the resolution, framerate, or datatype being set for the RGB sensor or the depth sensor. Typically, very finicky
values, I recommend leaving as is unless you find a spreadsheet with alternative specifications.