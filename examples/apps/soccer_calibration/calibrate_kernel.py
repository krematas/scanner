import scannerpy
import cv2
import numpy as np
import examples.apps.soccer_calibration.utils as utils


# A kernel file defines a standalone Python kernel which performs some computation by exporting a
# Kernel class.
class MyCalibrateKernel(scannerpy.Kernel):
    # __init__ is called once at the creation of the pipeline. Any arguments passed to the kernel
    # are provided through a protobuf object that you manually deserialize. See resize.proto for the
    # protobuf definition.
    def __init__(self, config, protobufs):
        self._w = config.args['w']
        self._h = config.args['h']
        self._A = config.args['A']
        self._R = config.args['R']
        self._T = config.args['T']

    # execute is the core computation routine maps inputs to outputs, e.g. here resizes an input
    # frame to a smaller output frame.
    def execute(self, columns):
        edge_sfactor = 0.5
        edges = utils.robust_edge_detection(cv2.resize(columns[0], None, fx=edge_sfactor, fy=edge_sfactor))
        edges = cv2.resize(edges, None, fx=1. / edge_sfactor, fy=1. / edge_sfactor)
        edges = cv2.Canny(edges.astype(np.uint8) * 255, 100, 200) / 255.0

        mask = cv2.dilate(columns[1][:, :, 0], np.ones((25, 25), dtype=np.uint8))

        edges = edges * (1 - mask)
        dist_transf = cv2.distanceTransform((1 - edges).astype(np.uint8), cv2.DIST_L2, 0)

        template, field_mask = utils.draw_field(self._A, self._R, self._T, self._h, self._w)

        II, JJ = (template > 0).nonzero()
        synth_field2d = np.array([[JJ, II]]).T[:, :, 0]

        field3d = utils.plane_points_to_3d(synth_field2d, self._A, self._R, self._T)

        self._A, self._R, self._T = utils.calibrate_camera_dist_transf(self._A, self._R, self._T, dist_transf, field3d)

        rgb = columns[0].copy()
        canvas, mask = utils.draw_field(self._A, self._R, self._T, self._h, self._w)
        canvas = cv2.dilate(canvas.astype(np.uint8), np.ones((15, 15), dtype=np.uint8)).astype(float)
        rgb = rgb * (1 - canvas)[:, :, None] + np.dstack((canvas * 255, np.zeros_like(canvas), np.zeros_like(canvas)))

        # result = np.dstack((template, template, template))*255

        out = rgb.astype(np.uint8)
        return [out]


KERNEL = MyCalibrateKernel
