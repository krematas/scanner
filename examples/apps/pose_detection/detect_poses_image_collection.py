import scannerpy
from scannerpy import Database, DeviceType, Job, FrameType
from scannerpy.stdlib import NetDescriptor, readers
import math
import os
import subprocess
import cv2
import sys
import os.path
import glob
from typing import Tuple
import argparse
from os.path import join
import time


# Testing settings
parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona/')
parser.add_argument('--visualize', action='store_true')

opt, _ = parser.parse_known_args()

dataset = opt.path_to_data
image_files = glob.glob(join(dataset, 'tmp', '*.jpg'))
image_files.sort()


db = Database()

encoded_image = db.sources.Files()
frame = db.ops.ImageDecoder(img=encoded_image)


sampler = db.streams.All
sampler_args = {}

if db.has_gpu():
    print('Using GPUs')
    device = DeviceType.GPU
    pipeline_instances = -1
else:
    print('Using CPUs')
    device = DeviceType.CPU
    pipeline_instances = 1

poses_out = db.ops.OpenPose(
    frame=frame,
    pose_num_scales=3,
    pose_scale_gap=0.33,
    device=device)


output_op = db.sinks.Column(columns={'frame': poses_out})

job = Job(
    op_args={
        encoded_image: {'paths': image_files},
        output_op: 'example_resized',
    })

print('Estimate poses')
start = time.time()
[out_table] = db.run(output_op, [job], force=True, work_packet_size=8, io_packet_size=64,
                     pipeline_instances_per_node=pipeline_instances)
end = time.time()

print('Total time for pose estimation in scanner: {0:.3f} sec'.format(end-start))
