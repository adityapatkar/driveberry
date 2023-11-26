import cv2
import numpy as np
import logging
import math
from tensorflow.keras.models import load_model

from src.cnn_driving.utility import img_preprocess, display_heading_line, show_image


class CNNDrive(object):
    def __init__(
        self, car=None, model_path="src/cnn_driving/model/lane_follower_cnn.keras"
    ):
        logging.info("Creating a CNNDrive instance...")

        self.car = car
        self.curr_steering_angle = 90
        self.model = load_model(model_path)

    def compute_steering_angle(self, frame):
        """
        Find the steering angle directly based on video frame
        We assume that camera is calibrated to point to dead center
        """
        preprocessed = img_preprocess(frame)
        x = np.asarray([preprocessed])
        steering_angle = self.model.predict(x)[0]

        logging.debug("new steering angle: %s" % steering_angle)
        return int(steering_angle + 0.5)

    def follow_lane(self, frame):
        # Main entry point of the lane follower
        show_image("orig", frame)

        self.curr_steering_angle = self.compute_steering_angle(frame)
        logging.debug("curr_steering_angle = %d" % self.curr_steering_angle)

        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)
        final_frame = display_heading_line(frame, self.curr_steering_angle)

        return final_frame
