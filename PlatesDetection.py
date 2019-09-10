import cv2
import math
import Preprocess2
import CharsDetection
import ClassChar

class ClassPlate:

    # constructor
    def __init__(Plate):
        Plate.imagePlate = None
        Plate.imageGrayscale = None
        Plate.imageThresh = None

        Plate.LocationOfPlateInScene = None

        Plate.strChars = ""

def findChars(imageThresh: object) -> object:
    listOfChars = []

    CountChars = 0

    imageThreshCopy = imageThresh.copy()

    contours, npaHierarchy = cv2.findContours(imageThreshCopy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(0, len(contours)):

        Char = ClassChar.Char(contours[i])

        if CharsDetection.checkIfChar(Char):                   # if contour is a possible char
            CountChars += 1
            listOfChars.append(Char)

    return listOfChars

def detectPlatesInImage(image):
    listOfPossiblePlates = []

    imageGrayscale, imageThresh = Preprocess2.preprocess(image)

            # find all possible chars,  this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
    listOfChars = findChars(imageThresh)

            # given a list of all possible chars, find groups of matching chars
    listOfListsOfMatchingChars = CharsDetection.findListOfListsOfChars(listOfChars)

    for listOfMatchingChars in listOfListsOfMatchingChars:
        possiblePlate = extractPlate(image, listOfMatchingChars)

        if possiblePlate.imagePlate is not None:
            listOfPossiblePlates.append(possiblePlate)

    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")  #

    return listOfPossiblePlates

def extractPlate(image, listOfChars):
    Plate = ClassPlate()

    listOfChars.sort(key = lambda matchingChar: matchingChar.CenterX)

            # calculate the center point of the plate
    PlateCenterX = (listOfChars[0].CenterX + listOfChars[len(listOfChars) - 1].CenterX) / 2.0
    PlateCenterY = (listOfChars[0].CenterY + listOfChars[len(listOfChars) - 1].CenterY) / 2.0

    PlateCenter = PlateCenterX, PlateCenterY

    # 1.15 represents the width padding of the plate box
    PlateWidth = int((listOfChars[len(listOfChars) - 1].BoundingRectX + listOfChars[len(listOfChars) - 1].BoundingRectWidth - listOfChars[0].BoundingRectX) * 1.15)

    TotalOfCharHeights = 0

    for Char in listOfChars:
        TotalOfCharHeights = TotalOfCharHeights + Char.BoundingRectHeight

    AverageCharHeight = TotalOfCharHeights / len(listOfChars)

    PlateHeight = int(AverageCharHeight * 1.35) # 1.35 represents the height padding of the plate box

            # calculate correction angle of region
    Opposite = listOfChars[len(listOfChars) - 1].CenterY - listOfChars[0].CenterY
    Hypotenuse = CharsDetection.distanceBetweenChars(listOfChars[0], listOfChars[len(listOfChars) - 1])
    CorrectionAngleRadians = math.asin(Opposite / Hypotenuse)
    CorrectionAngleDegrees = CorrectionAngleRadians * (180.0 / math.pi)

            # pack region center point, width, height, and correction angle into rotated rect variable of plate

    Plate.rrLocationOfPlateInScene = ( tuple(PlateCenter), (PlateWidth, PlateHeight), CorrectionAngleDegrees )

            # get the rotation matrix for our calculated correction angle

    rotationMatrix = cv2.getRotationMatrix2D(tuple(PlateCenter), CorrectionAngleDegrees, 1.0)

    height, width, numChannels = image.shape      # unpack original image width and height

    imageRotated = cv2.warpAffine(image, rotationMatrix, (width, height))       # rotate the entire image

    imageCropped = cv2.getRectSubPix(imageRotated, (PlateWidth, PlateHeight), tuple(PlateCenter))

    Plate.imagePlate = imageCropped

    return Plate