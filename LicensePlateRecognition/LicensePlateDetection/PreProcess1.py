import cv2
import numpy as np


#---- Increasing the contrast of the image--------------

#-----Reading the image-----------------------------------------------------
img = cv2.imread('C:/Users/Cosmin/AppData/Local/Programs/Python/Python36/LicensePlateRecognition/LicensePlateDetection/image_detected/Image2.jpg')
cv2.imshow("img",img)

#-----Converting image to LAB Color model-----------------------------------
lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("lab",lab)

#-----Splitting the LAB image to different channels-------------------------
l, a, b = cv2.split(lab)
cv2.imshow('l_channel', l)
cv2.imshow('a_channel', a)
cv2.imshow('b_channel', b)

#-----Applying CLAHE to L-channel-------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
cv2.imshow('CLAHE output', cl)

#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg = cv2.merge((cl,a,b))
cv2.imshow('limg', limg)

#-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imshow('final', final)

final1 = cv2.GaussianBlur(final,(7,7),cv2.BORDER_DEFAULT)
cv2.imshow('final1', final1)

#--------------- GrayScale----------------------------

im_gray = cv2.cvtColor(final1, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',im_gray)

#---------------- Binarize------------------------

(thresh, im_binarized) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow('binary',im_binarized)


#------------------  Remove Noise---------------------
cv2.imshow('2', thresh)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
kernel2 = np.ones((3,3),np.uint8)
dilation = cv2.dilate(im_binarized,kernel2,iterations = 7)
erosion = cv2.erode(dilation,kernel1,iterations = 1)

contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
c = max(contours, key = cv2.contourArea)
rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(im_binarized,[box],0,(0,0,255),2)

coords = np.column_stack(np.where(im_binarized > 0))
angle = cv2.minAreaRect(coords)[-1]
if angle < -45:
	angle = -(90 + angle)
else:
	angle = -angle

# rotate the image to deskew it
(h, w) = im_binarized.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(im_binarized, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

cv2.imwrite('C:/Users/Cosmin/AppData/Local/Programs/Python/Python36/LicensePlateRecognition/PlateRecognition/images/image9.1.jpg', rotated)

# show the output image
print("[INFO] angle: {:.3f}".format(angle))
cv2.imshow("Input", img)
cv2.imshow("Rotated", rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()
