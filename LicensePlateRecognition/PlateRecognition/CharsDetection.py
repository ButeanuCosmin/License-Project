import cv2
import numpy as np
import math
import Preprocess2
import ClassChar

kNearest = cv2.ml.KNearest_create()

def loadAndTrainKNN():

    try:
        TrainingClassifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print("error, unable to open classifications.txt\n")
        return False

    try:
        FlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print("error, unable to open flattened_images.txt\n")
        return False

    npaClassifications = TrainingClassifications.reshape((TrainingClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

    kNearest.setDefaultK(1)

    kNearest.train(FlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           # train KNN object

    return True

def checkIfChar(Char):

    # this function check on a contour to see if it could be a char,

    MIN_ASPECT_RATIO = 0.25
    MAX_ASPECT_RATIO = 1.0

    # 2 represents the pixel width, 8 represents the pixel height and 80 represents the pixel area. All of them can be modified for getting better results

    if (Char.BoundingRectArea > 80 and
        Char.BoundingRectWidth > 2 and Char.BoundingRectHeight > 8 and
        MIN_ASPECT_RATIO < Char.AspectRatio and Char.AspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False

# use Pythagorean theorem to calculate distance between two chars
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.CenterX - secondChar.CenterX)
    intY = abs(firstChar.CenterY - secondChar.CenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))

# use basic trigonometry to calculate angle between chars
def angleBetweenChars(firstChar, secondChar):
    Adj = float(abs(firstChar.CenterX - secondChar.CenterX))
    Opp = float(abs(firstChar.CenterY - secondChar.CenterY))

    if Adj != 0.0:
        AngleRadians = math.atan(Opp / Adj)
    else:
        AngleRadians = 1.5708                          # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program

    AngleDegrees = AngleRadians * (180.0 / math.pi)       # calculate angle in degrees

    return AngleDegrees

def findCharsInPlate(imgGrayscale, imageThresh):
    listOfPossibleChars = []
    imageThreshCopy = imageThresh.copy()

    contours, npaHierarchy = cv2.findContours(imageThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        Char = ClassChar.Char(contour)

        if checkIfChar(Char):
            listOfPossibleChars.append(Char)

    return listOfPossibleChars

def findListOfChars(Char, listOfChars):

    MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

            #  this function  gives a possible char and a big list of possible chars,
            # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
    listOfPossibleChars = []

    for possibleChar in listOfChars:
        if possibleChar == Char:    # if the char we attempting to find matches the exact same char as the char in the big list we are currently checking
                                                    # then we should not include it in the list of matches cause that would end up double including the current char
            continue

        DistanceBetweenChars = distanceBetweenChars(Char, possibleChar)

        AngleBetweenChars = angleBetweenChars(Char, possibleChar)

        ChangeInArea = float(abs(possibleChar.BoundingRectArea - Char.BoundingRectArea)) / float(Char.BoundingRectArea)

        ChangeInWidth = float(abs(possibleChar.BoundingRectWidth - Char.BoundingRectWidth)) / float(Char.BoundingRectWidth)
        ChangeInHeight = float(abs(possibleChar.BoundingRectHeight - Char.BoundingRectHeight)) / float(Char.BoundingRectHeight)

                # check if chars match
        # 0.5 represents the change in area, 0.8 the change in width and 0.2 the change in height. All of them can be modified
        if (DistanceBetweenChars < (Char.DiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            AngleBetweenChars < 12.0 and
            ChangeInArea < 0.5 and
            ChangeInWidth < 0.8 and
            ChangeInHeight < 0.2):
            listOfPossibleChars.append(possibleChar)        # if the chars are a match, add the current char to list of matching chars

    return listOfPossibleChars

def findListOfListsOfChars(listOfChars):

            # this function re-arrange the  big list of chars into a list of lists of matching chars,

    listOfListsOfChars = []
    CharsInPlate = 4
    for Char in listOfChars:
        listOfMatchingChars = findListOfChars(Char, listOfChars)        # find all chars in the big list that match the current char

        listOfMatchingChars.append(Char)                # add the current char to current possible list of matching chars

        if len(listOfMatchingChars) < CharsInPlate:     # if current possible list of matching chars is not long enough to constitute a possible plate
            continue                            # jump back to the top of the for loop and try again with next char, note that it's not necessary

                                                # if we get here, the current list passed test as a "group" or "cluster" of matching chars
        listOfListsOfChars.append(listOfMatchingChars)      # so add to our list of lists of matching chars

                                                # remove the current list of matching chars from the big list so we don't use the same chars twice,

        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfChars(listOfPossibleCharsWithCurrentMatchesRemoved)      # recursive call

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        # for each list of matching chars found by recursive call
            listOfListsOfChars.append(recursiveListOfMatchingChars)             # add to our original list of lists of matching chars

        break
    return listOfListsOfChars

def detectCharsInPlates(listOfPossiblePlates):

    if len(listOfPossiblePlates) == 0:
        return listOfPossiblePlates

    for possiblePlate in listOfPossiblePlates:

        possiblePlate.imageGrayscale, possiblePlate.imageThresh = Preprocess2.preprocess(possiblePlate.imagePlate)

        possiblePlate.imageThresh = cv2.resize(possiblePlate.imageThresh, (0, 0), fx = 1.5, fy = 1.5)

        thresholdValue, possiblePlate.imageThresh = cv2.threshold(possiblePlate.imageThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


                # this function first finds all contours and includes contours that could be chars
        listOfCharsInPlate = findCharsInPlate(possiblePlate.imageGrayscale, possiblePlate.imageThresh)

        listOfListsCharsInPlate = findListOfListsOfChars(listOfCharsInPlate)

        if (len(listOfListsCharsInPlate) == 0):

            possiblePlate.strChars = ""
            continue

        for i in range(0, len(listOfListsCharsInPlate)):
            listOfListsCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.CenterX)

                # within each possible plate, suppose the longest list of potential matching chars is the actual list of chars
        LenLongestListOfChars = 0
        IndexLongestListOfChars = 0

                # loop through all the vectors of matching chars, get the index of the one with the most chars
        for i in range(0, len(listOfListsCharsInPlate)):
            if len(listOfListsCharsInPlate[i]) > LenLongestListOfChars:
                LenLongestListOfChars = len(listOfListsCharsInPlate[i])
                IndexLongestListOfChars = i

        longestListOfCharsInPlate = listOfListsCharsInPlate[IndexLongestListOfChars]

        possiblePlate.strChars = recognizeChars(possiblePlate.imageThresh, longestListOfCharsInPlate)

    return listOfPossiblePlates

# this is where we apply the actual char recognition
def recognizeChars(imageThresh, listOfMatchingChars):

    SCALAR_GREEN = (0.0, 255.0, 0.0)
    strChars = ""

    ResisedImageWidth = 20
    ResisedImageHeight = 30

    height, width = imageThresh.shape

    imageThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.CenterX)

    cv2.cvtColor(imageThresh, cv2.COLOR_GRAY2BGR, imageThreshColor)

    for currentChar in listOfMatchingChars:
        pt1 = (currentChar.BoundingRectX, currentChar.BoundingRectY)
        pt2 = ((currentChar.BoundingRectX + currentChar.BoundingRectWidth), (currentChar.BoundingRectY + currentChar.BoundingRectHeight))

        cv2.rectangle(imageThreshColor, pt1, pt2, SCALAR_GREEN, 2)
        cv2.imshow('char', imageThreshColor)

                # crop char out of threshold image
        imageROI = imageThresh[currentChar.BoundingRectY : currentChar.BoundingRectY + currentChar.BoundingRectHeight,
                           currentChar.BoundingRectX : currentChar.BoundingRectX + currentChar.BoundingRectWidth]

        imageROIResized = cv2.resize(imageROI, (ResisedImageWidth, ResisedImageHeight))

        npaROIResized = imageROIResized.reshape((1, ResisedImageWidth * ResisedImageHeight))        # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)               # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)              # call findNearest !!!

        strCurrentChar = str(chr(int(npaResults[0][0])))            # get character from results

        strChars = strChars + strCurrentChar
    return strChars
