import cv2
import sys

from src.opencv_auto.driver import AutoDrive


def save_image_and_steering_angle(filename):
    """
    Save images and steering angle from a video file
    """
    lane_follower = AutoDrive()
    cap = cv2.VideoCapture(f"{filename}.avi")

    try:
        i = 0
        while cap.isOpened():
            _, frame = cap.read()
            lane_follower.follow_lane(frame)
            cv2.imwrite(
                f"{filename}_{i:03d}_{lane_follower.curr_steering_angle:03d}.png",
                frame,
            )
            i += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    save_image_and_steering_angle(sys.argv[1])
