import cv2
import numpy as np
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter


def load_labels(path):
    with open(path, "r") as file:
        return {i: line.strip() for i, line in enumerate(file.readlines())}


def draw_objects(image, results, labels):
    for obj in results:
        # Draw the bounding box
        bbox = obj.bbox
        cv2.rectangle(
            image, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2
        )

        # Draw label and score
        label = f"{labels[obj.id]} {obj.score:.2f}"
        cv2.putText(
            image,
            label,
            (bbox.xmin, bbox.ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    # Display the image
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_objects(image_path, model_path, label_path):
    # Load the TFLite model and allocate tensors.
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    # Load labels
    labels = load_labels(label_path)

    # Read and preprocess an image.
    image = cv2.imread(image_path)
    _, scale = common.set_resized_input(
        interpreter, image.shape[:2], lambda size: cv2.resize(image, size)
    )

    # Run inference
    interpreter.invoke()

    # Get detection results
    results = detect.get_objects(interpreter, score_threshold=0.4, image_scale=scale)

    # Draw the results on the image
    draw_objects(image, results, labels)


# Example usage
model_path = (
    "src/object_detection/artifacts/model.tflite"  # Path to the TFLite model file
)
label_path = "src/object_detection/artifacts/labels.txt"  # Path to the labels file
image_path = "src/object_detection/artifacts/image.png"  # Path to the image file

detect_objects(image_path, model_path, label_path)
