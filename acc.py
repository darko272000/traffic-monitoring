import cv2
import numpy as np
import torch
import time

# import custom YOLO 5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="accident.pt")
# load input image or video stream


def acc_detection(frame):
    """
    Detects and extracts information about traffic signs in a given frame.

    Args:
        frame: The input frame to perform traffic sign detection on.

    Returns:
        A tuple containing the label of the detected traffic sign, as well as the coordinates of its bounding box.

    """
    results = model(frame, size=640)
    # filter predictions to conf > 0.5
    results = results.pandas().xyxy[0][results.pandas().xyxy[0]["confidence"] > 0.6]
    # draw bounding boxes around detected traffic signs
    for i in range(len(results)):
        x1, y1, x2, y2 = results.iloc[i][["xmin", "ymin", "xmax", "ymax"]].astype(int)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # draw label
        labelt = results.iloc[i]["name"]
        # cv2.putText(frame, labelt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # print confidence
        conf = round(results.iloc[i]["confidence"], 2)
        # cv2.putText(frame, str(conf), (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return labelt, x1, y1, x2, y2

    # show frame

    # press q to exit
