import scannerpy
import cv2
import numpy as np

from scannerpy import Database, DeviceType, Job, ColumnType, FrameType
from scannerpy.stdlib import pipelines

import subprocess
from os.path import join
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from hourglass import hg8
from dataset_loader import get_set


@scannerpy.register_python_op()
class MyDepthEstimationClass(scannerpy.Kernel):
    def __init__(self, config):



        # self.transforms = transforms.Compose([Rescale(opt.img_size, opt.label_size), ToTensor(), NormalizeImage()])

        self.logsoftmax = nn.LogSoftmax()


        self._w = config.args['w']
        self._h = config.args['h']
        self.net = config.args['model']

    def execute(self, image: FrameType, mask: FrameType) -> FrameType:
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        image = image.transpose((2, 0, 1))/255.0
        mask = mask[:, :, None].transpose((2, 0, 1))

        image_tensor = torch.from_numpy(image)
        image_tensor = torch.FloatTensor(image_tensor.size()).copy_(image_tensor)
        mask_tensor = torch.from_numpy(mask)

        input, mask = Variable(image_tensor).float(), Variable(mask_tensor).float()

        input = torch.cat((input, mask), 1)
        input = input.cuda()

        output = self.net(input)
        final_prediction = self.logsoftmax(output[-1])

        return cv2.resize((image*(mask/255.)).astype(np.uint8), (self._w, self._h))


dataset = '/home/krematas/Mountpoints/grail/data/barcelona/'
image_files = glob.glob(join(dataset, 'players', 'images', '*.jpg'))
image_files.sort()
image_files = image_files[:10]

mask_files = glob.glob(join(dataset, 'players', 'masks', '*.png'))
mask_files.sort()
mask_files = mask_files[:10]


db = Database()

encoded_image = db.sources.Files()
frame = db.ops.ImageDecoder(img=encoded_image)

encoded_mask = db.sources.Files()
mask_frame = db.ops.ImageDecoder(img=encoded_mask)

checkpoint = torch.load('/home/krematas/Mountpoints/grail/tmp/cnn/model.pth')
netG_state_dict = checkpoint['state_dict']
netG = hg8(input_nc=4, output_nc=51)
netG.load_state_dict(netG_state_dict)
netG.cuda()


my_depth_estimation_class = db.ops.MyDepthEstimationClass(image=frame, mask=mask_frame,  w=60, h=60, model=netG)
output_op = db.sinks.FrameColumn(columns={'frame': my_depth_estimation_class})

job = Job(
    op_args={
        encoded_image: {'paths': image_files},
        encoded_mask: {'paths': mask_files},

        output_op: 'example_resized',
    })

[out_table] = db.run(output_op, [job], force=True)
out_table.column('frame').save_mp4('haha')
