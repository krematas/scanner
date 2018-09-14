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

mask_files = glob.glob(join(dataset, 'players', 'masks', '*.png'))
mask_files.sort()

db = Database()

encoded_image = db.sources.Files()
frame = db.ops.ImageDecoder(img=encoded_image)

encoded_mask = db.sources.Files()
frame2 = db.ops.ImageDecoder(img=encoded_mask)

my_resize_imageset_class = db.ops.MyResizeClass(image=frame, mask=frame2,  w=60, h=60)
output_op = db.sinks.FrameColumn(columns={'frame': my_resize_imageset_class})

job = Job(
    op_args={
        encoded_image: {'paths': ['/home/krematas/Mountpoints/grail/data/barcelona/players/images/00114_00010.jpg',
                                  '/home/krematas/Mountpoints/grail/data/barcelona/players/images/00114_00008.jpg']},
        encoded_mask: {'paths': ['/home/krematas/Mountpoints/grail/data/barcelona/players/masks/00114_00010.png',
                                  '/home/krematas/Mountpoints/grail/data/barcelona/players/masks/00114_00008.png']},

        output_op: 'example_resized',
    })
[out_table] = db.run(output_op, [job], force=True)
out_table.column('frame').save_mp4('haha')
