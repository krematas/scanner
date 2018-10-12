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

@scannerpy.register_python_op(name='PoseDraw')
def pose_draw(self, frame: FrameType, frame_poses: bytes) -> FrameType:
    for pose in readers.poses(frame_poses, self.protobufs):
        pose.draw(frame)
    return frame


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
start = time.time()
[out_table] = db.run(output_op, [job], force=True)
end = time.time()





# drawn_frame = db.ops.PoseDraw(frame=frame, frame_poses=poses_out)
#
#
#
#
#
#
# sampled_frames = sampler(drawn_frame)
# output = db.sinks.Column(columns={'frame': sampled_frames})
# job = Job(
#     op_args={
#         frame: input_table.column('frame'),
#         sampled_frames: sampler_args,
#         output: movie_name + '_drawn_poses',
# })
# [drawn_poses_table] = db.run(output=output, jobs=[job], work_packet_size=8, io_packet_size=64,
#                              pipeline_instances_per_node=pipeline_instances,
#                              force=True)
#
# print('Writing output video...')
# drawn_poses_table.column('frame').save_mp4('{:s}_poses'.format(movie_name))
