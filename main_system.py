from flask import Flask, Response, render_template, send_file, jsonify ,make_response
import cv2
from ultralytics import YOLO
import cv2
import math
import yaml
from trafficlight import *
from acc import *
import cvzone
from adjust import *
import base64
from datetime import datetime, timedelta

with open('polygons.yml', 'r') as f:
     polygons = yaml.load(f, Loader=yaml.FullLoader)
vi = None
global cars
cars = set()
global violation
violation = set()
global statu
statu = None
global accframe
accframe = None
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)


# Use the device index of your webcam or the file path of a video file
# For example, to use the default webcam, use 0
# To use a video file, use its file path, such as "path/to/video.mp4"


def gen_frames():
    """
    Generator function that processes frames from a video and yields them as byte strings.

    Returns:
        byte string: The processed frame in the form of a byte string.
    """
    global acc
    global vi 
    global cars
    global violation
    global statu
    global accframe

    cars.clear()
    violation=set()
    area = polygons[0]
    area2 = polygons[1]

    model = YOLO('yolov8s.pt')
    results = model.track('Untitled video - Made with Clipchamp.mp4' ,stream=True,verbose=False, agnostic_nms=True)
    class_names2=['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    class_names = model.model.names
    z=24

    for r in results:
        z=z+1
        frame = r.orig_img
        shape= frame.shape
        boxes = r.boxes

        if z % 25 == 0:
            status = traffic(frame)
            acc = acc_detection(frame)

        if type(status) == tuple:
            statu = status[0]
            x1_l, y1_l, x2_l, y2_l = status[1], status[2], status[3], status[4]
            cv2.rectangle(frame, (x1_l, y1_l), (x2_l, y2_l), (0, 255, 0), 2)
            cv2.putText(frame, statu, (x1_l, y1_l),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if type(acc) == tuple:
            ac = acc[0]
            x1_a, y1_a, x2_a, y2_a = acc[1], acc[2], acc[3], acc[4]
            w_a, h_a = x2_a - x1_a, y2_a - y1_a
            cvzone.cornerRect(frame, (x1_a, y1_a, w_a, h_a),
                              l=5, rt=0, t=2, colorR=(255, 0, 255),)
            cvzone.putTextRect(frame, ac, (max(0, x1_a), max(
                35, y1_a)), scale=0.3, thickness=1, offset=10, font=cv2.FONT_HERSHEY_SIMPLEX)
            cv2.imwrite("acc.jpg", frame)
            accframe = frame[y1_a-100:y2_a+100, x1_a-100:x2_a+100]

        frame = cv2.resize(frame, (1280, 720))   

        for box in boxes:
            if box.id != None:
                id =box.id.cpu().numpy().astype(int)[0]
                cls = int(box.cls[0])
                class_name = class_names[cls]

                if class_name  in class_names2:
                    x1, y1, w, h =adjust_bbox(box.xyxy[0], shape)

                    if cv2.pointPolygonTest(np.array(area, np.int32), (w, h), False)==1:
                        c, b = w-x1, h-y1
                        cars.add(id)
                        cvzone.cornerRect(frame, (x1, y1, c, b), l=5,
                                          rt=0, t=2, colorR=(255, 0, 255),)
                        cvzone.putTextRect(frame, str(id), (max(0, x1), max(35, y1)),
                                           scale=0.3, thickness=1, offset=10, font=cv2.FONT_HERSHEY_SIMPLEX)

                    if cv2.pointPolygonTest(np.array(area2, np.int32), (w, h), False)==1:
                        cars.clear()
                        cvzone.cornerRect(frame, (x1, y1, c, b), l=5, rt=0, t=2, colorC=(
                            0, 0, 255), colorR=(255, 0, 255))

                        if statu == 'Red':
                            cvzone.putTextRect(frame, 'violation', (max(0, x1), max(35, y1)),
                                               scale=0.5, thickness=1, offset=10, font=cv2.FONT_HERSHEY_SIMPLEX)
                            violation.add(id)
                            frame2 = frame[y1-100:h+100, x1-100:w+100]
                            filename2=f"vi.jpg"
                            cv2.imwrite(filename2, frame2)
                            vi=frame2

                conf = math.ceil((box.conf[0] * 100)) / 100
               #  cv2.putText(frame, 'the number of cars is :'+str(len(cars)),
                           # (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
               # cv2.putText(frame, 'the number of violation is :'+str(len(violation)),
                           # (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

    


@app.route('/')
def index():
   
    return render_template('index2.html',vi=len(cars))


@app.route('/video_feed')
def video_feed():
    """
    Returns a video feed as a response.

    This function calls the gen_frames() function to generate frames and returns them as a response with the mimetype 'multipart/x-mixed-replace; boundary=frame'.

    Returns:
        Response: The video feed response.
    """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/get_value')
def get_value(cars=cars):
    """
    Retrieves the values related to cars, violations, and status.

    Returns:
        A JSON response containing the values of cars, violations, and status.
    """
    # Your code that generates a changing value goes here
    value = len(cars)
    value2 = len(violation)
    value3 = statu
    value = str(value)
    value2 = str(value2)
    value3 = str(value3)
    values = [value, value2, value3]
    return jsonify(values)



@app.route('/violations')
def violations():
    """
    Endpoint for handling violations.

    Returns:
        Response: The response containing the image base64 data and timestamp.
    """
    global vi
    if vi is None:
        # create a black image
        return Response('no violations', mimetype='text/plain')
    else:
        vi = cv2.resize(vi, (220, 220))
        _, image_data = cv2.imencode('.jpg', vi)
        image_base64 = base64.b64encode(image_data).decode('ascii')
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
        response = make_response(jsonify({'image_base64': image_base64, 'timestamp': timestamp}))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Expires'] = 0
        response.headers['Pragma'] = 'no-cache'
        vi = None
        return response

@app.route('/accidents')
def accidents():
    """
    Endpoint for retrieving accident images.

    Returns:
        Response: A response object containing the accident image in base64 format and the timestamp.
    """
    global accframe
    if accframe is None:
        # create a black image
        return Response('no accidents', mimetype='text/plain')
    else:
        accframe = cv2.resize(accframe, (220, 220))
        _, image_data = cv2.imencode('.jpg', accframe)
        image_base64 = base64.b64encode(image_data).decode('ascii')
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
        response = make_response(jsonify({'image_base64': image_base64, 'timestamp': timestamp}))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Expires'] = 0
        response.headers['Pragma'] = 'no-cache'
        accframe = None
        return response
app.route('/prames')

if __name__ == '__main__':
    app.run()