import cv2
from src.object_detection.model import EdgeTPUModel
import logging
import time
import yaml


class Detector(object):
    def __init__(
        self,
        car=None,
        model_file="src/object_detection/artifacts/yolov5s-int8-224_edgetpu.tflite",
        names_file="src/object_detection/artifacts/coco.yaml",
        width=640,
        height=480,
        speed_limit=35,
    ):
        self.model_file = model_file
        self.names_file = names_file
        self.model = EdgeTPUModel(model_file, names_file)
        self.car = car
        self.width = width
        self.height = height
        self.speed_limit = speed_limit

        # read yaml file for names
        with open(names_file) as f:
            self.names = yaml.load(f, Loader=yaml.FullLoader)["names"]
            self.labels = {}
            for i, name in enumerate(self.names):
                self.labels[i] = name

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10, height - 10)
        self.fontScale = 1
        self.fontColor = (255, 255, 255)  # white
        self.boxColor = (0, 0, 255)  # RED
        self.boxLineWidth = 1
        self.lineType = 2
        self.annotate_text = ""
        self.annotate_text_time = time.time()
        self.time_to_show_prediction = 1.0

    def detect_objects(self, frame):
        start_ms = time.time()
        detections = self.model.predict(frame)
        if detections:
            for obj in detections:
                height = obj.bounding_box[1][1] - obj.bounding_box[0][1]
                width = obj.bounding_box[1][0] - obj.bounding_box[0][0]
                logging.debug(
                    "%s, %.0f%% w=%.0f h=%.0f"
                    % (self.labels[obj.label_id], obj.score * 100, width, height)
                )
                box = obj.bounding_box
                coord_top_left = (int(box[0][0]), int(box[0][1]))
                coord_bottom_right = (int(box[1][0]), int(box[1][1]))
                cv2.rectangle(
                    frame,
                    coord_top_left,
                    coord_bottom_right,
                    self.boxColor,
                    self.boxLineWidth,
                )
                annotate_text = "%s %.0f%%" % (
                    self.labels[obj.label_id],
                    obj.score * 100,
                )
                coord_top_left = (coord_top_left[0], coord_top_left[1] + 15)
                cv2.putText(
                    frame,
                    annotate_text,
                    coord_top_left,
                    self.font,
                    self.fontScale,
                    self.boxColor,
                    self.lineType,
                )
        else:
            logging.debug("No objects detected")

        elapsed_ms = time.time() - start_ms

        annotate_summary = "%.1f FPS" % (1.0 / elapsed_ms)
        logging.debug(annotate_summary)
        cv2.putText(
            frame,
            annotate_summary,
            self.bottomLeftCornerOfText,
            self.font,
            self.fontScale,
            self.fontColor,
            self.lineType,
        )
        # cv2.imshow('Detected Objects', frame)

        return detections, frame

    def control_car(self, objects):
        logging.debug("Control car...")

        if len(objects) == 0:
            logging.debug(
                "No objects detected, drive at speed limit of %s." % self.speed_limit
            )

        for obj in objects:
            obj_label = self.labels[obj.label_id]
            if obj_label == "Stop":
                logging.debug("Stop sign detected, stopping car.")
                self.set_speed(0)
                time.sleep(3)
                self.set_speed(self.speed_limit)
                break

    def set_speed(self, speed):
        self.speed = speed
        if self.car is not None:
            logging.debug("Actually setting car speed to %d" % speed)
            self.car.back_wheels.speed = speed

    def process_objects_on_road(self, frame):
        # Main entry point of the Road Object Handler
        logging.debug("Processing objects.................................")
        objects, final_frame = self.detect_objects(frame)
        self.control_car(objects)
        logging.debug("Processing objects END..............................")

        return final_frame
