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
from transforms import *


@scannerpy.register_python_op()
class MyDepthEstimationClass(scannerpy.Kernel):
    def __init__(self, config):

        checkpoint = torch.load(opt.modelpath)
        netG_state_dict = checkpoint['state_dict']
        self.netG = hg8(input_nc=4, output_nc=51)
        self.netG.load_state_dict(netG_state_dict)
        self.netG.cuda()

        self.logsoftmax = nn.LogSoftmax()


        self._w = config.args['w']
        self._h = config.args['h']

    def execute(self, image: FrameType, mask: FrameType) -> FrameType:

        input, mask = Variable(image).float(), Variable(mask).float()

        input = torch.cat((input, mask), 1)
        input = input.cuda()

        output = self.netG(input)
        final_prediction = self.logsoftmax(output[-1])

        return cv2.resize((image*(mask/255.)).astype(np.uint8), (self._w, self._h))
