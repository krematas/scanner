import scannerpy
import cv2
import numpy as np

from scannerpy import Database, DeviceType, Job, ColumnType, FrameType

from os.path import join
import glob

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

from hourglass import hg8
import matplotlib.pyplot as plt
from scipy.misc import imresize


@scannerpy.register_python_op()
class MyDepthEstimationClass(scannerpy.Kernel):
    def __init__(self, config):

        checkpoint = torch.load('/home/krematas/Mountpoints/grail/tmp/cnn/model.pth')
        netG_state_dict = checkpoint['state_dict']
        netG = hg8(input_nc=4, output_nc=51)
        netG.load_state_dict(netG_state_dict)
        netG.cuda()

        self.logsoftmax = nn.LogSoftmax()
        self.normalize = transforms.Normalize(mean=[0.3402085, 0.42575407, 0.23771574],
                                         std=[0.1159472, 0.10461029, 0.13433486])

        self.img_size = config.args['img_size']
        self.net = netG

    def execute(self, image: FrameType, mask: FrameType) -> FrameType:

        # Rescale
        image = imresize(image, (self.img_size, self.img_size))
        mask = imresize(mask[:, :, 0], (self.img_size, self.img_size), interp='nearest', mode='F')

        # ToTensor
        image = image.transpose((2, 0, 1))/255.0
        mask = mask[:, :, None].transpose((2, 0, 1))/255.0

        image_tensor = torch.from_numpy(image)
        image_tensor = torch.FloatTensor(image_tensor.size()).copy_(image_tensor)
        mask_tensor = torch.from_numpy(mask)

        # Normalize
        image_tensor = self.normalize(image_tensor)

        # Make it BxCxHxW
        image_tensor = image_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)

        # Concat input and mask
        image_tensor = torch.cat((image_tensor.float(), mask_tensor.float()), 1)
        image_tensor = image_tensor.cuda()

        output = self.net(image_tensor)
        final_prediction = self.logsoftmax(output[-1])

        np_prediction = final_prediction.cpu().detach().numpy()
        np_prediction = np_prediction[0, :, :, :]

        return np_prediction.astype(np.float32)


dataset = '/home/krematas/Mountpoints/grail/data/barcelona/'
image_files = glob.glob(join(dataset, 'players', 'images', '*.jpg'))
image_files.sort()
image_files = image_files[:20]

mask_files = glob.glob(join(dataset, 'players', 'masks', '*.png'))
mask_files.sort()
mask_files = mask_files[:20]

pred_files = glob.glob(join(dataset, 'players', 'predictions', '*.npy'))
pred_files.sort()
pred_files = pred_files[:20]


db = Database()

encoded_image = db.sources.Files()
frame = db.ops.ImageDecoder(img=encoded_image)

encoded_mask = db.sources.Files()
mask_frame = db.ops.ImageDecoder(img=encoded_mask)


my_depth_estimation_class = db.ops.MyDepthEstimationClass(image=frame, mask=mask_frame, img_size=256)
output_op = db.sinks.FrameColumn(columns={'frame': my_depth_estimation_class})

job = Job(
    op_args={
        encoded_image: {'paths': image_files},
        encoded_mask: {'paths': mask_files},

        output_op: 'example_resized',
    })

[out_table] = db.run(output_op, [job], force=True)

results = out_table.column('frame').load()


for i, res in enumerate(results):
    pred = np.load(pred_files[i])[0, :, :, :]
    pred = np.argmax(pred, axis=0)
    fig, ax = plt.subplots(1, 2)

    pred_scanner = np.argmax(res, axis=0)

    ax[1].imshow(pred)
    ax[0].imshow(pred_scanner)
    plt.show()
# out_table.column('frame').save_mp4('haha')
