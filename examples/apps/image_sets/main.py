import scannerpy
from scannerpy import Database, DeviceType, Job, ColumnType
import cv2


def b(s):
    return bytes(s, 'utf-8')


cwd = '/home/krematas/code/scanner/examples/apps/image_sets'

db = Database()


encoded_image = db.sources.Files()
frame = db.ops.ImageDecoder(img=encoded_image)

encoded_mask = db.sources.Files()
frame2 = db.ops.ImageDecoder(img=encoded_mask)

db.register_op('ProcessImage', [('image', ColumnType.Video), ('mask', ColumnType.Video)], [('resized', ColumnType.Video)])
db.register_python_kernel('ProcessImage', DeviceType.CPU, cwd + '/dummy_kernel.py')

resized = db.ops.ProcessImage(image=frame, mask=frame2,  w=60, h=60)
output_op = db.sinks.FrameColumn(columns={'frame': resized})

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

