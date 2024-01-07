from ultralytics import YOLO
import cv2
import math
model = YOLO('light.pt' )


def traffic(img):
    """
    Detects traffic lights in an image and returns the class name and bounding box coordinates.

    Args:
        img (numpy.ndarray): The input image.

    Returns:
        tuple: A tuple containing the class name, x1, y1, x2, y2 coordinates of the bounding box.
    """
    results = model(img, verbose=False, agnostic_nms=True)
    class_names = ['Green', 'off', 'Red', 'Yallow']
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            return class_names[cls], x1, y1, x2, y2
