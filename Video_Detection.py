
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config',
     help = 'path to yolo config file', default='/path/to/yolov3-tiny.cfg')
ap.add_argument('-w', '--weights',
    help = 'path to yolo pre-trained weights', default='/path/to/yolov3-tiny_finally.weights')
ap.add_argument('-cl', '--classes',
    help = 'path to text file containing class names',default='/path/to/objects.names')
args = ap.parse_args()

def draw_pred(video, class_id, confidence, x, y, x_plus_w, y_plus_h):

    SCALAR_RED = (0.0, 0.0, 255.0)
    SCALAR_GREEN = (0.0, 255.0, 0.0)
    label = "license plate"

    cv2.rectangle(video, (x,y), (x_plus_w,y_plus_h), SCALAR_RED ,2)

    cv2.putText(video, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SCALAR_GREEN,2)

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromDarknet(args.config, args.weights)

# determine only the *output* layer names that we need from YOLO
layernames = net.getLayerNames()
layernames = [layernames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
videoStream = cv2.VideoCapture('C:/Users/Cosmin/AppData/Local/Programs/Python/Python36/LicensePlateRecognition/LicensePlateDetection/videos/video1.mp4')
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(videoStream.get(prop))
    print("[INFO] {} total frames in video".format(total))

except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = videoStream.read()

    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward pass of the YOLO object detector, giving us our bounding boxes and associated probabilities

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(layernames)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    count=0
    # loop over each of the layer outputs
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5 :

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

                #while count < total:
                if confidence > 0.9:
                    crop = frame[y:y + H, x:x + W]
                    cv2.imwrite(
                            "C:/Users/Cosmin/AppData/Local/Programs/ython/Python36/LicensePlateRecognition/LicensePlateDetection/image_detected/frame%d.jpg" % count,
                            crop)
                    count += 1

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(indices) > 0:

        for i in indices.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            draw_pred(frame, classIDs[i], confidences[i], round(x), round(y), round(x + w), round(y + h))


    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter('videos/video1_output.avi', fourcc, 30,
            (frame.shape[1], frame.shape[0]), True)

        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))

    # write the output frame to disk
    writer.write(frame)

print("[INFO] cleaning up...")
writer.release()
videoStream.release()