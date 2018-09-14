import scannerpy
import cv2
import numpy as np
from scannerpy import Database, DeviceType, Job, ColumnType, FrameType
from scannerpy.stdlib import pipelines

import subprocess
from os.path import join
import numpy as np
import glob
import os


dataset = '/home/krematas/Mountpoints/grail/data/barcelona/'
image_files = glob.glob(join(dataset, 'players', 'images', '*.jpg'))
image_files.sort()
image_files = image_files[:100]

mask_files = glob.glob(join(dataset, 'players', 'masks', '*.png'))
mask_files.sort()
mask_files = mask_files[:100]

db = Database()

cwd = os.path.dirname(os.path.abspath(__file__))
if not os.path.isfile(os.path.join(cwd, 'resize_op/build/libresize_op.so')):
    print(
        'You need to build the custom op first: \n'
        '$ pushd {}/segment_op; mkdir build && cd build; cmake ..; make; popd'.
        format(cwd))
    exit()

# To load a custom op into the Scanner runtime, we use db.load_op to open the
# shared library we compiled. If the op takes arguments, it also optionally
# takes a path to the generated python file for the arg protobuf.
db.load_op(
    os.path.join(cwd, 'segment_op/build/libsegment_op.so'),
    os.path.join(cwd, 'segment_op/build/resize_pb2.py'))



encoded_image = db.sources.Files()
frame = db.ops.ImageDecoder(img=encoded_image)

encoded_mask = db.sources.Files()
mask_frame = db.ops.ImageDecoder(img=encoded_mask)

my_resize_imageset_class = db.ops.MyResizeClass(image=frame, mask=mask_frame,  w=60, h=60)
output_op = db.sinks.FrameColumn(columns={'frame': my_resize_imageset_class})

job = Job(
    op_args={
        encoded_image: {'paths': image_files},
        encoded_mask: {'paths': mask_files},

        output_op: 'example_resized',
    })
[out_table] = db.run(output_op, [job], force=True)
out_table.column('frame').save_mp4('haha')
