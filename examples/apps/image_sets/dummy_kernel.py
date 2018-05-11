import scannerpy
import cv2


class MyDummyKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self._w = config.args['w']
        self._h = config.args['h']

    def execute(self, columns):
        return [cv2.resize(columns[0], (self._w, self._h))]


KERNEL = MyDummyKernel
