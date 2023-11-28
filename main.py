import logging
import datetime
import picar
import cv2

from src.opencv_auto.driver import AutoDrive

# from src.cnn_driving.driver import CNNDrive
from src.object_detection.model import DetectionModel
from src.opencv_auto.utility import show_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriveBerry(object):
    """
    Base class for a self-driving car
    """

    def __init__(self, initial_speed=35, screen_width=640, screen_height=480):
        logging.info("Creating an instance of DriveBerry")

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.initial_speed = initial_speed

        picar.setup()

        logging.debug("Setting up camera")
        self.camera = cv2.VideoCapture(-1)
        self.camera.set(
            3, self.screen_width
        )  # this will set the width of the camera to 320 pixels
        self.camera.set(4, self.screen_height)

        logging.debug("Setting up front wheels")
        self.front_wheels = picar.front_wheels.Front_Wheels()

        logging.debug("Setting up back wheels")
        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.back_wheels.forward()
        self.back_wheels.speed = 0

        self.lane_follower = AutoDrive(self)
        # self.lane_follower = CNNDrive(self)

        self.object_detector = DetectionModel(self)

        logging.debug("Setting up video capture")

        self.fourcc = cv2.VideoWriter_fourcc(
            *"XVID"
        )  # this is the codec that works for .avi
        datestr = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.video_orig = self.create_video_recorder(f"./data/car_video{datestr}.avi")
        self.video_lane = self.create_video_recorder(
            f"./data/car_video_lane{datestr}.avi"
        )
        self.video_objs = self.create_video_recorder(
            f"./data/car_video_objs{datestr}.avi"
        )

    def create_video_recorder(self, filename):
        """
        Create a video recorder with the given filename
        """
        return cv2.VideoWriter(
            filename, self.fourcc, 20.0, (self.screen_width, self.screen_height)
        )

    def __enter__(self):
        """Entering a with statement"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit a with statement"""
        if traceback is not None:
            # Exception occurred:
            logging.error("Exiting with statement with exception %s" % traceback)

        self.cleanup()

    def cleanup(self):
        """Reset the hardware"""
        logging.info("Stopping the car, resetting hardware.")
        self.back_wheels.speed = 0
        self.front_wheels.turn(90)
        self.camera.release()
        self.video_orig.release()
        self.video_lane.release()
        self.video_objs.release()
        cv2.destroyAllWindows()

    def drive(self, speed=35):
        """
        Drive the car at a given speed
        """
        logging.info(f"Starting to drive at speed {speed}...")
        self.back_wheels.speed = speed
        i = 0
        while self.camera.isOpened():
            ret, lane_frame = self.camera.read()
            object_frame = lane_frame.copy()

            if ret:
                i += 1
                logging.debug(f"Processing frame {i}")
                self.video_orig.write(lane_frame)

                # show_image("Detected Objects", object_frame)

                # object_frame = self.process_objects_on_road(object_frame)
                # self.video_objs.write(object_frame)

                lane_frame = self.lane_follower.follow_lane(lane_frame)
                self.video_lane.write(lane_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    def follow_lane(self, image):
        """
        Main entry point of the lane follower
        """
        image = self.lane_follower.follow_lane(image)
        return image

    def process_objects_on_road(self, image):
        image = self.object_detector.process_objects_on_road(image)
        return image


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting car")
    with DriveBerry() as car:
        car.drive(35)
