import scannerpy
import cv2
import numpy as np

class MyDummyKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self._w = config.args['w']
        self._h = config.args['h']

    def execute(self, columns):
        return [cv2.resize((columns[0]*(columns[1]/255.)).astype(np.uint8), (self._w, self._h))]


KERNEL = MyDummyKernel
