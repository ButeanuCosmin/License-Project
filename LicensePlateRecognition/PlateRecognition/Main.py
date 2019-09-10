import cv2
import os
import CharsDetection
import PlatesDetection

def main():

    Training = CharsDetection.loadAndTrainKNN()

    if Training == False:
        print("\nerror: KNN traning was not successful\n")
        return

    image  = cv2.imread("images/image5.1.jpg")

    if image is None:
        print("image not read from file")
        os.system("pause")
        return

    listOfPlates = PlatesDetection.detectPlatesInImage(image)

    listOfPlates = CharsDetection.detectCharsInPlates(listOfPlates)

    cv2.imshow("image", image)

    if len(listOfPlates) == 0:
        print("no license plates were detected")
    else:

                # sort the list of  plates in DESCENDING order (most number of chars to least number of chars)
        listOfPlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # suppose the plate with the most recognized chars is the actual plate
        licensePlate = listOfPlates[0]

        cv2.imshow("imagePlate", licensePlate.imagePlate)
        cv2.imshow("imageThresh", licensePlate.imageThresh)

        if len(licensePlate.strChars) == 0:
            print("no characters were detected")
            return

        drawRedRectangleAroundPlate(image, licensePlate)

        print("\nlicense plate from image = " + licensePlate.strChars + "\n")

        cv2.imshow("image", image)

    cv2.waitKey(0)
    return

def drawRedRectangleAroundPlate(image, licPlate):

    SCALAR_RED = (0.0, 0.0, 255.0)
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect

    cv2.line(image, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
    cv2.line(image, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(image, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(image, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

if __name__ == "__main__":
    main()
