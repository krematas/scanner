from scannerpy import Database, DeviceType, Job, ColumnType
from scannerpy.stdlib import pipelines

import subprocess
import cv2
import sys
import os.path
import examples.util
import numpy as np

movie_path = '/home/krematas/Mountpoints/grail/data/barcelona/test.mp4'
print('Detecting faces in movie {}'.format(movie_path))
movie_name = os.path.splitext(os.path.basename(movie_path))[0]

mask_path = '/home/krematas/Mountpoints/grail/data/barcelona/mask.mp4'
db = Database()
print('Ingesting video into Scanner ...')
input_tables, failed = db.ingest_videos([(movie_name, movie_path), ('mask', mask_path)], force=True)

print(db.summarize())
print('Failures:', failed)

cam_data = np.load('/home/krematas/Mountpoints/grail/data/barcelona/calib/00114.npy').item()


db.register_op('Calibrate', [('frame', ColumnType.Video), ('mask', ColumnType.Video)], [('resized', ColumnType.Video)])

# Custom Python kernels for ops reside in a separate file, here calibrate_kernel.py.
cwd = '/home/krematas/code/scanner/examples/apps/soccer_calibration'
db.register_python_kernel('Calibrate', DeviceType.CPU, cwd + '/calibrate_kernel.py')

frame = db.sources.FrameColumn()
mask = db.sources.FrameColumn()

# Then we use our op just like in the other examples.
resized = db.ops.Calibrate(frame=frame, mask=mask, w=3840, h=2160, A=cam_data['A'], R=cam_data['R'], T=cam_data['T'])
output_op = db.sinks.FrameColumn(columns={'frame': resized})

job = Job(op_args={
    frame: input_tables[0].column('frame'),
    mask: input_tables[1].column('frame'),
    output_op: 'example_resized',
})
[out_table] = db.run(output_op, [job], force=True)
out_table.column('frame').save_mp4(movie_name + '_faces')

print('Successfully generated {:s}_faces.mp4'.format(movie_name))





#
# sampler = db.sampler.all()
#
#
#
#
#
# print('Detecting faces...')
# [bboxes_table] = pipelines.detect_faces(
#     db, [input_table.column('frame')], sampler,
#     movie_name + '_bboxes')
#
# print('Drawing faces onto video...')
# frame = db.sources.FrameColumn()
# sampled_frame = frame.sample()
# bboxes = db.sources.Column()
# out_frame = db.ops.DrawBox(frame = sampled_frame, bboxes = bboxes)
# output = db.sinks.Column(columns={'frame': out_frame})
# job = Job(op_args={
#     frame: input_table.column('frame'),
#     sampled_frame: sampler,
#     bboxes: bboxes_table.column('bboxes'),
#     output: movie_name + '_bboxes_overlay',
# })
# [out_table] = db.run(output=output, jobs=[job], force=True)
# out_table.column('frame').save_mp4(movie_name + '_faces')
#
# print('Successfully generated {:s}_faces.mp4'.format(movie_name))
