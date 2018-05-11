import scannerpy
from scannerpy import Database, DeviceType, Job, ColumnType
import cv2


def b(s):
    return bytes(s, 'utf-8')


db = Database()

img1 = cv2.imread('/home/krematas/Mountpoints/grail/data/barcelona/players/images/00114_00010.jpg')
img2 = cv2.imread('/home/krematas/Mountpoints/grail/data/barcelona/players/images/00114_00008.jpg')
cwd = '/home/krematas/code/scanner/examples/apps/image_sets'


db.new_table('mine', ['index', 'image'], [[b('00114_00010'), img1.tobytes()], [b('00114_00008'), img2.tobytes()]], force=True)
input_table = db.table('mine')

db.register_op('ProcessImage', [('image', ColumnType.Video)], [('resized', ColumnType.Video)])
db.register_python_kernel('ProcessImage', DeviceType.CPU, cwd + '/dummy_kernel.py')


frame = db.sources.FrameColumn()


# Then we use our op just like in the other examples.
resized = db.ops.ProcessImage(frame=frame, w=60, h=60)
output_op = db.sinks.FrameColumn(columns={'frame': resized})

job = Job(op_args={
    frame: input_table.column('image'),
    output_op: 'example_resized',
})
[out_table] = db.run(output_op, [job], force=True)
out_table.column('frame').save_mp4('haha')

