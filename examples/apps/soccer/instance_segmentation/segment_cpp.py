import scannerpy
import cv2
import numpy as np
from scannerpy import Database, DeviceType, Job, ColumnType, FrameType
from scannerpy.stdlib import pipelines
import time

from os.path import join
import glob
import os

import argparse


parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona/')
parser.add_argument('--visualize', action='store_true')
opt, _ = parser.parse_known_args()


dataset = opt.path_to_data
image_files = glob.glob(join(dataset, 'players', 'images', '*.jpg'))
image_files.sort()

mask_files = glob.glob(join(dataset, 'players', 'poseimgs', '*.png'))
mask_files.sort()

db = Database()

cwd = os.path.dirname(os.path.abspath(__file__))
if not os.path.isfile(os.path.join(cwd, 'segment_op/build/libsegment_op.so')):
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
    os.path.join(cwd, 'segment_op/build/segment_pb2.py'))


encoded_image = db.sources.Files()
frame = db.ops.ImageDecoder(img=encoded_image)

encoded_mask = db.sources.Files()
mask_frame = db.ops.ImageDecoder(img=encoded_mask)

my_segment_imageset_class = db.ops.MySegment(frame=frame, mask=mask_frame, w=128, h=128, sigma1=1.0, sigma2=0.01)
output_op = db.sinks.FrameColumn(columns={'frame': my_segment_imageset_class})

job = Job(
    op_args={
        encoded_image: {'paths': image_files},
        encoded_mask: {'paths': mask_files},

        output_op: 'example_resized',
    })

start = time.time()
[out_table] = db.run(output_op, [job], force=True)
end = time.time()

print('Total time for instance segmentation in scanner: {0:.3f} sec'.format(end-start))
# out_table.column('frame').save_mp4(join(dataset, 'players', 'instance_segm.mp4'))
