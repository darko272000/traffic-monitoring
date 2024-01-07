from trafficlight import *
from adjust import *
from acc import *
from ultralytics import YOLO
import PyQt5
from qt_material import apply_stylesheet
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtCore import *
import yaml
import torch
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtGui as QtGui
import numpy as np
import cvzone
import cv2
import sys

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# from sort import *
# import missing modules
# from trafic import *

global points


class VideoThread(QThread):
    """
    A QThread subclass for processing video frames and emitting signals.

    Attributes:
        change_pixmap_signal (pyqtSignal): Signal emitted when a new frame is processed.
        video_finished_signal (pyqtSignal): Signal emitted when the video is finished.

    Methods:
        __init__(): Initializes the VideoThread object.
        run(): Runs the video processing loop.
        stop(): Stops the video processing loop.
    """
    change_pixmap_signal = pyqtSignal(np.ndarray)
    video_finished_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        """
        Runs the video processing loop.

        This method reads frames from the video, processes them, and emits the change_pixmap_signal
        with the processed frame. It also checks for the stop flag to gracefully stop the loop.
        """
        self.cap = cv2.VideoCapture(input_path)
        numberofframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        model = YOLO("yolov8m.pt")

        with open("polygons.yml", "r") as f:
            polygons = yaml.load(f, Loader=yaml.FullLoader)

        area = polygons[0]

        area2 = polygons[1]
        violation = set()
        cars = set()
        # results = model.track('y3.mp4' ,stream=True)
        class_names2 = ["car", "truck", "bus",
                        "train", "motorcycle", "bicycle"]
        class_names = model.model.names
        z = 19
        framenumber = 0
        statu = None

        results = model.track(
            input_path, stream=True, verbose=False, persist=True, tracker="botsort.yaml")

        while self._run_flag == True:
            ret, test = self.cap.read()
            if not ret:
                self.video_finished_signal.emit()
                self._run_flag = False
                self.cap.release()
                break

            for r in results:
                z = z + 1
                framenumber = framenumber + 1
                frame = r.orig_img
                shape = frame.shape
                boxes = r.boxes
                # print('frame number',framenumber)
                if self._run_flag == False:
                    self.cap.release()
                    break
                if z % 20 == 0:
                    status = traffic(frame)
                    acc = acc_detection(frame)

                if type(status) == tuple:
                    statu = status[0]
                    x1_l, y1_l, x2_l, y2_l = status[1], status[2], status[3], status[4]
                    cv2.rectangle(frame, (x1_l, y1_l),
                                  (x2_l, y2_l), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        statu,
                        (x1_l, y1_l),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                if type(acc) == tuple:
                    ac = acc[0]
                    x1_a, y1_a, x2_a, y2_a = acc[1], acc[2], acc[3], acc[4]
                    w_a, h_a = x2_a - x1_a, y2_a - y1_a
                    cvzone.cornerRect(
                        frame,
                        (x1_a, y1_a, w_a, h_a),
                        l=5,
                        rt=0,
                        t=2,
                        colorR=(255, 0, 255),
                    )
                    cvzone.putTextRect(
                        frame,
                        ac,
                        (max(0, x1_a), max(35, y1_a)),
                        scale=0.3,
                        thickness=1,
                        offset=10,
                        font=cv2.FONT_HERSHEY_SIMPLEX,
                    )
                    # save the acc
                    cv2.imwrite("acc.jpg", frame)
                frame = cv2.resize(frame, (1280, 720))

                smoothed_polygon = cv2.approxPolyDP(
                    np.array(area, np.int32), 7, True)
                smoothed_polygon = cv2.convexHull(smoothed_polygon)
                # s
                # cv2.polylines(frame, [smoothed_polygon], True, (0, 255, 0), 2)
                smoothed_polygon2 = cv2.approxPolyDP(
                    np.array(area2, np.int32), 7, True)
                smoothed_polygon2 = cv2.convexHull(smoothed_polygon2)
                # cv2.polylines(frame, [smoothed_polygon2], True, (0, 255, 0), 2)

                for box in boxes:
                    if box.id != None:
                        id = box.id.cpu().numpy().astype(int)[0]
                        cls = int(box.cls[0])
                        class_name = class_names[cls]

                        if class_name in class_names2:
                            x1, y1, w, h = adjust_bbox(box.xyxy[0], shape)

                            if (
                                cv2.pointPolygonTest(
                                    np.array(area, np.int32), (w, h), False
                                )
                                == 1
                            ):
                                c, b = w - x1, h - y1
                                cars.add(id)
                                cvzone.cornerRect(
                                    frame,
                                    (x1, y1, c, b),
                                    l=5,
                                    rt=0,

                                    t=2,
                                    colorR=(255, 0, 255),
                                )
                                cvzone.putTextRect(
                                    frame,
                                    str(id),
                                    (max(0, x1), max(35, y1)),
                                    scale=0.3,
                                    thickness=1,
                                    offset=10,
                                    font=cv2.FONT_HERSHEY_SIMPLEX,
                                )
                            # cv2.putText(frame, f'{class_name}', (x1, y1 - 25),
                            # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            if (
                                cv2.pointPolygonTest(
                                    np.array(area2, np.int32), (w, h), False
                                )
                                == 1
                            ):
                                cars.clear()
                                cvzone.cornerRect(
                                    frame,
                                    (x1, y1, c, b),
                                    l=5,
                                    rt=0,
                                    t=2,
                                    colorC=(0, 0, 255),
                                    colorR=(255, 0, 255),
                                )
                                if statu == "Red":
                                    cvzone.putTextRect(
                                        frame,
                                        "violation",
                                        (max(0, x1), max(35, y1)),
                                        scale=0.5,
                                        thickness=1,
                                        offset=10,
                                        font=cv2.FONT_HERSHEY_SIMPLEX,
                                    )
                                    violation.add(id)
                                    # save the car img
                                    print(violation)
                                    frame2 = frame[y1 - 100: h +
                                                   100, x1 - 100: w + 100]
                                    cv2.imwrite("car.jpg", frame2)
                        # Confidence
                        conf = math.ceil((box.conf[0] * 100)) / 100
                cv2.putText(
                    frame,
                    "the number of cars is :" + str(len(cars)),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "the number of violation is :" + str(len(violation)),
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                self.change_pixmap_signal.emit(frame)

    def stop(self):
        """
        Stops the video processing loop.

        Sets the run flag to False and waits for the thread to finish.
        """
        self._run_flag = False


class MyWindow(QMainWindow):
    """
    Represents a custom window for displaying and interacting with video.

    Attributes:
        thread (VideoThread): The video capture thread.
        frame (QFrame): The frame widget for displaying the video.
        title_label (QLabel): The label widget for displaying the title.
        pro_label (QLabel): The label widget for displaying the processing status.
        ss_video (QPushButton): The button widget for starting/stopping the video.
        ss_video2 (QPushButton): The button widget for choosing a video.
        ss_video3 (QPushButton): The button widget for choosing parameters.
        ss_video4 (QRadioButton): The radio button widget for selecting live video.
        status (QStatusBar): The status bar widget for displaying messages.
        image_label (QLabel): The label widget for displaying the video frames.

    Methods:
        __init__(): Initializes the MyWindow object.
        initWindow(): Initializes the window layout and widgets.
        ClickStartVideo(): Activates when the Start/Stop video button is clicked to start the video.
        ClickStopVideo(): Activates when the Start/Stop video button is clicked to stop the video.
        chosevideo(): Activates when the choose video button is clicked.
        parmeter(): Activates when the choose parameters button is clicked.
        hide_select_video(): Hides the select video button and disables other buttons.
        show_select_video(): Shows the select video button and enables other buttons.
        update_image(cv_img): Updates the image_label with a new OpenCV image.
        convert_cv_qt(cv_img): Converts an OpenCV image to QPixmap.
    """

    def __init__(self):
        super(MyWindow, self).__init__()
        # Getting available cameras

        # Finds the center of the screen
        cent = QDesktopWidget().availableGeometry().center()
        #  self.setStyleSheet("background-color: white;")
        self.resize(1310, 900)
        self.frameGeometry().moveCenter(cent)
        self.setWindowTitle("A I")
        self.initWindow()

    ########################################################################################################################
    #                                                   Windows                                                            #
    ########################################################################################################################
    def initWindow(self):
        # create the video capture thread
        self.thread = VideoThread()

        # Label with the name of the co-founders
        self.frame = QtWidgets.QFrame(self)
        self.frame.setStyleSheet("border: 2px solid teal;")
        self.frame.resize(1290, 740)

        self.frame.move(10, 20)

        # Label with the name of the co-founders
        self.title_label = QtWidgets.QLabel(self)  # Create label
        self.title_label.setObjectName("title")

        # Add text to label
        self.title_label.move(23, 10)
        self.title_label.resize(40, 20)  # Set size for the label
        self.title_label.setStyleSheet("background-color: #31363b; ")
        self.title_label.setText("Video")
        self.title_label.setAlignment(
            Qt.AlignCenter)  # Align text in the label
        self.pro_label = QtWidgets.QLabel(self.frame)
        self.pro_label.setAlignment(Qt.AlignCenter)

        self.pro_label.setStyleSheet("border: none;")
        self.pro_label.resize(200, 70)
        x = (self.frame.width() - self.pro_label.width()) // 2
        y = (self.frame.height() - self.pro_label.height()) // 2
        self.pro_label.move(x, y)

        self.ss_video = QtWidgets.QPushButton(self)
        self.ss_video.setText("Start video")
        self.ss_video.move(855, 780)
        self.ss_video.resize(200, 70)
        self.ss_video.clicked.connect(self.ClickStartVideo)
        self.ss_video.setEnabled(False)
        self.ss_video2 = QtWidgets.QPushButton(self)
        self.ss_video2.setText("chose video")
        self.ss_video2.move(540, 780)
        self.ss_video2.resize(200, 70)
        self.ss_video2.clicked.connect(self.chosevideo)
        self.ss_video3 = QtWidgets.QPushButton(self)
        self.ss_video3.setText("chose prametras")
        self.ss_video3.move(220, 780)
        self.ss_video3.resize(200, 70)
        self.ss_video3.clicked.connect(self.parmeter)
        self.ss_video3.setEnabled(False)
        self.ss_video4 = QtWidgets.QRadioButton(self)
        self.ss_video4.setText("live video")
        self.ss_video4.move(1150, 780)
        self.ss_video4.resize(200, 70)
        self.ss_video4.clicked.connect(self.hide_select_video)

        # Status bar
        self.status = QStatusBar()
        # self.status.setStyleSheet("background : lightblue;")  # Setting style sheet to the status bar
        self.setStatusBar(self.status)  # Adding status bar to the main window
        self.status.showMessage("Ready to start")

        self.image_label = QLabel(self.frame)
        self.image_label.setStyleSheet("border: none; ")
        self.disply_width = 1270
        self.display_height = 720

        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.move(10, 10)

    ########################################################################################################################
    #                                                   Buttons                                                            #
    ########################################################################################################################
    # Activates when Start/Stop video button is clicked to Start (ss_video

    def ClickStartVideo(self):
        # Change label color to light blue
        self.ss_video.clicked.disconnect(self.ClickStartVideo)
        self.status.showMessage("Video Running...")
        # Change button to stop
        self.ss_video.setText("Stop video")
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.video_finished_signal.connect(self.ClickStopVideo)
        self.pro_label.setText("Processing...")

        # start the thread
        self.thread.start()
        # Stop the video if button clicked
        self.ss_video.clicked.connect(self.thread.stop)
        self.ss_video.clicked.connect(self.ClickStopVideo)

    # Activates when Start/Stop video button is clicked to Stop (ss_video)
    def ClickStopVideo(self):
        self.thread.change_pixmap_signal.disconnect()

        self.ss_video.setText("Start video")
        self.status.showMessage("Ready to start")
        self.ss_video.clicked.disconnect(self.ClickStopVideo)
        self.ss_video.clicked.disconnect(self.thread.stop)
        self.ss_video.clicked.connect(self.ClickStartVideo)
        self.pro_label.clear()
        self.image_label.clear()

    def chosevideo(self):
        # chose the video path
        global input_path
        self.path = QFileDialog.getOpenFileName(
            self, "Open File", "C:\\", "Video Files (*.mp4 *.avi *.flv *.mkv)"
        )[0]
        # save the video path to input_path variable
        input_path = self.path
        self.ss_video.setEnabled(True)
        self.ss_video3.setEnabled(True)

    def parmeter(self):
        print(self.path)
        parmeters(self.path)

    def hide_select_video(self):
        self.ss_video2.hide()
        global input_path
        self.path = 0
        self.ss_video.setEnabled(True)
        self.ss_video3.setEnabled(True)
        self.ss_video4.clicked.connect(self.show_select_video)

    def show_select_video(self):
        self.ss_video2.show()
        global input_path
        input_path = 0
        self.ss_video.setEnabled(False)
        self.ss_video3.setEnabled(False)
        self.ss_video4.clicked.connect(self.hide_select_video)
    ########################################################################################################################
    #                                                   Actions                                                            #
    ########################################################################################################################

    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(
            self.disply_width, self.display_height, Qt.KeepAspectRatio
        )
        # p = convert_to_Qt_format.scaled(801, 801, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


def parmeters(video_path):
    """
    Function to handle the parameters for drawing polygons on an image.

    Args:
        video_path (str): The path to the video file.

    Returns:
        None
    """
    poly_points = []
    points = []
    drawing = False
    polem = []

    # Rest of the code...
def parmeters(video_path):
    poly_points = []
    points = []
    drawing = False
    polem = []

    # This is the mouse callback function
    def mouse_callback(event, x, y, flags, param):
        global poly_points, points, drawing

        # If the left mouse button is pressed, start drawing the polymer
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            poly_points = [(x, y)]
            points = []

        # If the left mouse button is released, finish drawing the polymer
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            poly_points.append((x, y))
            # cv2.polylines(img, [np.array(poly_points)], True, (0, 255, 0), 2)
            points.append(poly_points)
            polem.append(points)
            poly_points = []
            with open("polygons.yml", "w") as f:
                yaml.dump(polem, f)

        # If the mouse is being moved and the left button is pressed, add points to the polymer
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            poly_points.append((x, y))

    # Load the video

    cap = cv2.VideoCapture(video_path)

    # take the 10th frame of the video
    for i in range(100):
        ret, img = cap.read()

    # Resize the image to a more manageable size
    img = cv2.resize(img, (1280, 720))

    # Create a window and bind the mouse callback function to it
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    # Main loop
    while True:
        # Copy the image so we can draw on it
        draw_img = img.copy()

        # Draw all the saved polygons on the image
        with open("polygons.yml", "r") as f:
            polygons = yaml.load(f, Loader=yaml.FullLoader)
            if polygons:
                for polygon in polygons:
                    polygon = np.array(polygon, dtype=np.int32)
                    smoothed_polygon = cv2.approxPolyDP(polygon, 40, True)
                    smoothed_polygon = cv2.convexHull(smoothed_polygon)
                    # s
                    cv2.polylines(
                        draw_img, [smoothed_polygon], True, (0, 255, 0), 2)

        # Show the image
        cv2.imshow("Image", draw_img)

        # Check for user input
        key = cv2.waitKey(1)
        if key == 27:  # Esc key
            break

    # Clean up
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_teal.xml")
    win = MyWindow()
    win.show()
    sys.exit(app.exec())
