
"""python Image_Detection.py -c C:/Users/Cosmin/Desktop/darknet-master/build/darknet/x64/yolov3-tiny-obj.cfg
-w C:/Users/Cosmin/Desktop/darknet-master/build/darknet/x64/yolov3-tiny.weights -cl C:/Users/Cosmin/Desktop/darknet-master/build/darknet/x64/data/obj.names"""

# import the necessary packages
import numpy as np
import argparse
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

def draw_pred(img, confidence, x, y, x_plus_w, y_plus_h):

	SCALAR_RED = (0.0, 0.0, 255.0)
	SCALAR_GREEN = (0.0, 255.0, 0.0)

	label = "LICENSE PLATE"

	#color = COLORS[class_id]

	cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), SCALAR_RED ,2)

	cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SCALAR_GREEN , 2)


# Load names classes
# classes = None
with open(args.classes, 'r') as f:
	classes = [line.strip() for line in f.readlines()]
print(classes)


# load our YOLO object detector trained
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNet(args.config, args.weights)


# load our input image and grab its spatial dimensions
image = cv2.imread('C:/Users/Cosmin/AppData/Local/Programs/Python/Python36/LicensePlateRecognition/LicensePlateDetection/images/img.jpg')
print(image)
image = cv2.resize(image,(1024,1024))
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
layernames = net.getLayerNames()
layernames = [layernames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward pass of the YOLO object detector, giving us our bounding boxes and associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(layernames)

boxes = []
confidences = []
classIDs = []
conf_threshold = 0.5
nms_threshold = 0.4

for output in layerOutputs:

	for detection in output:

		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		if confidence > 0.5 :

			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences, and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)
			crop = image[y:y+height,x:x+width]
			cv2.imwrite('C:/Users/Cosmin/AppData/Local/Programs/Python/Python36/LicensePlateRecognition/LicensePlateDetection/image_detected/Image2.jpg', crop)
			cv2.imshow("crop", crop)

# apply non-maxima suppression to suppress weak, overlapping bounding boxes

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

if len(indices) > 0:

	for i in indices.flatten():

		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
		draw_pred(image, confidences[i], round(x), round(y), round(x + w), round(y + h))

cv2.imshow("Image", image)
cv2.waitKey(0)
