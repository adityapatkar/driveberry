import os
import logging
import time

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter

import cv2

logger = logging.getLogger(__name__)


class DetectionModel(object):
    def __init__(
        self,
        car=None,
        speed_limit=35,
        model_path="src/object_detection/artifacts/model_edgetpu.tflite",
        label_path="src/object_detection/artifacts/labels.txt",
    ):
        logger.info("Initializing DetectionModel")
        self.car = car
        self.speed_limit = speed_limit
        self.model_path = model_path
        self.label_path = label_path
        self.width = 640
        self.height = 480
        self.labels = self.load_labels(self.label_path)

    def load_labels(self, path):
        with open(path, "r") as file:
            return {i: line.strip() for i, line in enumerate(file.readlines())}

    def detect_objects(self, frame):
        # Load the TFLite model and allocate tensors.
        interpreter = make_interpreter(self.model_path)
        interpreter.allocate_tensors()

        # Load labels
        labels = self.labels

        # Read and preprocess an image.
        _, scale = common.set_resized_input(
            interpreter, frame.shape[:2], lambda size: cv2.resize(frame, size)
        )

        # Run inference
        interpreter.invoke()

        # Get detection results
        results = detect.get_objects(
            interpreter, score_threshold=0.4, image_scale=scale
        )

        if results:
            for obj in results:
                bbox = obj.bbox
                height = bbox.ymin - bbox.ymax
                width = bbox.xmin - bbox.xmax
                logger.debug(
                    "%s, %.0f%% w=%.0f h=%.0f"
                    % (labels[obj.id], obj.score * 100, width, height)
                )

                cv2.rectangle(
                    frame,
                    (bbox.xmin, bbox.ymin),
                    (bbox.xmax, bbox.ymax),
                    (0, 255, 0),
                    2,
                )

                label = f"{labels[obj.id]} {obj.score:.2f}"
                print(label)

        else:
            logging.debug("No objects detected")

        return results, frame

    def is_close_by(self, obj, frame_height, min_height_pct=0.05):
        # default: if a sign is 10% of the height of frame
        bbox = obj.bbox
        obj_height = bbox.ymin - bbox.ymax
        return obj_height / frame_height > min_height_pct

    def control_car(self, objects):
        logger.debug("Controlling car")
        if len(objects) == 0:
            logger.debug("No objects detected, continue driving")
            for obj in objects:
                label = self.labels[obj.id]
                if label == "stop sign":
                    logger.info("Detected Stop Sign")
                    if self.is_close_by(obj, self.height):
                        self.car.back_wheels.speed = 0
                        time.sleep(3)
                        self.car.back_wheels.speed = 35

    def process_objects_on_road(self, frame):
        # Main entry point of the Road Object Handler
        logging.debug("Processing objects.................................")
        objects, final_frame = self.detect_objects(frame)
        self.control_car(objects)
        logging.debug("Processing objects END..............................")

        return final_frame
