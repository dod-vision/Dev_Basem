"""
You need first to install https://github.com/ildoonet/tf-pose-estimation
and add the subdirectory src to PYTHONPATH

to use the model:
tfOpenpose    = tf_openpose_detector.tfOpenpose()
tfOpenpose.detect(image)
"""

import argparse
import logging
import time

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)



class tfOpenpose:
    def __init__(self, zoom = 1.0, resolution = '656x368', model = 'cmu', show_process = False):
        self.zoom = zoom
        self.resolution = resolution
        self.model = model
        self.show_process = show_process
        logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
        self.w, self.h = model_wh(resolution)
        self.e = TfPoseEstimator(get_graph_path(model), target_size=(self.w, self.h))


    def detect(self, image):
        logger.debug('image preprocess+')
        if self.zoom < 1.0:
            canvas = np.zeros_like(image)
            img_scaled = cv2.resize(image, None, fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            image = canvas
        elif self.zoom > 1.0:
            img_scaled = cv2.resize(image, None, fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

        logger.debug('image process+')
        humans = self.e.inference(image)

        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        fps_time = 0

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        logger.debug('finished+')
