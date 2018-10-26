from scannerpy import Database, Job, DeviceType, FrameType
import scannerpy
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

################################################################################
# This tutorial shows how to write and use new Ops in Python.                  #
################################################################################

# Let's say we want to resize the frames in a video. If Scanner doesn't come
# with a built in "resize" op, we'll have to make our own. There are two ways to
# make a Python op (we also refer to Python ops as "kernels"). The simplest way
# is to write a function that will run independently over each element of the
# input sequence. For example:

# Ops have to be registered with the Scanner runtime, which is done
# here using the decorator @scannerpy.register_python_op()
@scannerpy.register_python_op()
def resize_fn(config, frame: FrameType) -> FrameType:
    # Function ops first input (here, config) is always the kernel config.
    # The kernel config provides metadata about the invocation of the Op,
    # such as:
    # - config.devices: What devices am I allocated to (CPU, GPU, # of devices)
    # - config.args: Arguments provided to the Op when specified in a
    #                computation graph
    # - etc

    # All other inputs must be annotated with one of two types: FrameType for
    # inputs which represent images, or bytes for inputs which represent
    # serialized values. Here we have only one input 'frame' which is of
    # type 'FrameType'.

    # The return type must also be annotated with the same types. For multiple
    # return values, you can specify a tuple of arguments. For example, if
    # we returned both an image and a serialized value, it would be:
    # -> Tuple[FrameType, bytes]

    # Here, we use the width and height from the config args to resize the
    # image.
    return cv2.resize(frame, (config.args['width'], config.args['height']))

# If your op has state (e.g. it tracks objects over time) or if it has high
# start-up costs (e.g. it loads a neural network model into memory), then you
# can also use our class-based interface:
@scannerpy.register_python_op()
class ResizeClass(scannerpy.Kernel):
    # Init runs once when the class instance is initialized
    def __init__(self, config):
        self._width = config.args['width']
        self._height = config.args['height']

    # The execute method serves the same purpose the registered op function
    # above does and has to provide the same type annotations.
    def execute(self, frame: FrameType) -> FrameType:
        return cv2.resize(frame, (self._width, self._height))

# Now we can use these new Ops in Scanner:
db = Database()

# Download an example video
example_video_path = util.download_video()

# Ingest it into the database
[input_table], _ = db.ingest_videos([('example', example_video_path)],
                                    force=True)

frame = db.sources.FrameColumn()

resized_frame_fn = db.ops.resize_fn(frame=frame, width=640, height=480)

resized_frame_class = db.ops.ResizeClass(frame=frame, width=320, height=240)

output = db.sinks.FrameColumn(columns={'frame1': resized_frame_fn,
                                       'frame2': resized_frame_class})

job = Job(op_args={
    frame: input_table.column('frame'),
    output: 'example_python_op'
})

[table] = db.run(output=output, jobs=[job], force=True)

table.column('frame1').save_mp4('01_resized_fn')
table.column('frame2').save_mp4('01_resized_class')

print('Finished! Two videos were saved to the current directory: '
      '01_resized_fn.mp4, 01_resized_class.mp4')

# If you are trying to integrate with a C++ library or you want a more efficient
# implementation for your Ops, you can also define Ops in C++. See the
# 08_defining_cpp_ops.py tutorial.
