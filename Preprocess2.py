import cv2

# module level variables
GAUSSIAN_SMOOTH_FILTER_SIZE = (13, 15)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

def preprocess(image):
    imageGrayscale = extractValue(image)

    imageMaxContrastGrayscale = maximizeContrast(imageGrayscale)

    height, width = imageGrayscale.shape

    #imaegBlurred = np.zeros((height, width, 1), np.uint8)

    #imageBlurred = cv2.GaussianBlur(imageMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    imageThresh = cv2.adaptiveThreshold(imageGrayscale, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return imageGrayscale, imageThresh

def extractValue(image):

    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    imageHue, imageSaturation, imageValue = cv2.split(imageHSV)

    return imageValue

def maximizeContrast(imageGrayscale):

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imageTopHat = cv2.morphologyEx(imageGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imageBlackHat = cv2.morphologyEx(imageGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imageGrayscalePlusTopHat = cv2.add(imageGrayscale, imageTopHat)
    imageGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imageGrayscalePlusTopHat, imageBlackHat)

    return imageGrayscalePlusTopHatMinusBlackHat