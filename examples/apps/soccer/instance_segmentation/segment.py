import scannerpy
import cv2
import numpy as np
from scannerpy import Database, DeviceType, Job, ColumnType, FrameType
from scannerpy.stdlib import pipelines

import subprocess
from os.path import join
import numpy as np
import glob


@scannerpy.register_python_op()
class MyResizeClass(scannerpy.Kernel):
    def __init__(self, config):
        self._w = config.args['w']
        self._h = config.args['h']

    def execute(self, image: FrameType, mask: FrameType) -> FrameType:
        return cv2.resize((image*(mask/255.)).astype(np.uint8), (self._w, self._h))


dataset = '/home/krematas/Mountpoints/grail/data/barcelona/'
image_files = glob.glob(join(dataset, 'players', 'images', '*.jpg'))
image_files.sort()
image_files = image_files[:100]

mask_files = glob.glob(join(dataset, 'players', 'masks', '*.png'))
mask_files.sort()
mask_files = mask_files[:100]

db = Database()

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
