import scannerpy
import cv2
from scannerpy import Database, DeviceType, Job, ColumnType, FrameType

from os.path import join
import numpy as np
import glob
import argparse
import pickle

import time

parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona/')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--cloud', action='store_true')
parser.add_argument('--bucket', default='', type=str)
opt, _ = parser.parse_known_args()


@scannerpy.register_python_op()
class DrawPosesClass(scannerpy.Kernel):
    def __init__(self, config):
        self.w = config.args['w']
        self.h = config.args['h']

        self.limps = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 11], [11, 12], [12, 13], [1, 8],
             [8, 9], [9, 10], [14, 15], [16, 17], [0, 14], [0, 15], [14, 16], [15, 17]])

    def execute(self, image: FrameType, poses: bytes) -> FrameType:

        poses = pickle.loads(poses)
        # output = np.zeros((self.h, self.w, 3), dtype=np.float32) - 1
        output = image
        for i in range(len(poses)):
            keypoints = poses[i]

            lbl = i
            for k in range(self.limps.shape[0]):
                kp1, kp2 = self.limps[k, :].astype(int)
                bone_start = keypoints[kp1, :]
                bone_end = keypoints[kp2, :]
                bone_start[0] = np.maximum(np.minimum(bone_start[0], self.w - 1), 0.)
                bone_start[1] = np.maximum(np.minimum(bone_start[1], self.h - 1), 0.)

                bone_end[0] = np.maximum(np.minimum(bone_end[0], self.w - 1), 0.)
                bone_end[1] = np.maximum(np.minimum(bone_end[1], self.h - 1), 0.)

                if bone_start[2] > 0.0:
                    output[int(bone_start[1]), int(bone_start[0])] = 1
                    cv2.circle(output, (int(bone_start[0]), int(bone_start[1])), 2, (lbl, 0, 0), -1)

                if bone_end[2] > 0.0:
                    output[int(bone_end[1]), int(bone_end[0])] = 1
                    cv2.circle(output, (int(bone_end[0]), int(bone_end[1])), 2, (lbl, 0, 0), -1)

                if bone_start[2] > 0.0 and bone_end[2] > 0.0:
                    cv2.line(output, (int(bone_start[0]), int(bone_start[1])), (int(bone_end[0]), int(bone_end[1])),
                             (lbl, 0, 0), 1)

        return output[:, :, 0]


dataset = opt.path_to_data


image_files = glob.glob(join(dataset, 'images', '*.jpg'))
image_files.sort()



db = Database()

config = db.config.config['storage']
params = {'bucket': opt.bucket,
          'storage_type': config['type'],
          'endpoint': 'storage.googleapis.com',
          'region': 'US'}

encoded_image = db.sources.Files(**params)
frame = db.ops.ImageDecoder(img=encoded_image)

with open(join(dataset, 'metadata', 'poses.p'), 'rb') as f:
    openposes = pickle.load(f)


draw_poses_class = db.ops.DrawPosesClass(image=frame, poses=openposes, h=2160, w=3840, device=DeviceType.CPU)
output_op = db.sinks.FrameColumn(columns={'frame': draw_poses_class})

job = Job(
    op_args={
        encoded_image: {'paths': image_files, **params},
        output_op: 'example_resized5',
    })

start = time.time()
[out_table] = db.run(output_op, [job], force=True, work_packet_size=8, io_packet_size=16)
out_table.column('frame').save_mp4(join(dataset, 'players', 'instance_segm.mp4'))

end = time.time()

print('Total time for depth estimation in scanner: {0:.3f} sec'.format(end-start))