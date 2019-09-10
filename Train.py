
import sys
import numpy as np
import cv2
import os

def main():

    RESIZED_IMAGE_WIDTH = 40
    RESIZED_IMAGE_HEIGHT = 50
    image = cv2.imread("train_images/monark_regular.png")

    if image is None:
        print ("error: image not read from file \n\n")
        os.system("pause")
        return

    Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Blurred = cv2.GaussianBlur(Gray, (5,5), 0)


    Thresh = cv2.adaptiveThreshold(Blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 3)

    cv2.imshow("Thresh", Thresh)
    ThreshCopy = Thresh.copy()

    Contours, Hierarchy = cv2.findContours(ThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    FlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    Classifications = []

    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for Contour in Contours:
        if cv2.contourArea(Contour) > 0:
            [intX, intY, intW, intH] = cv2.boundingRect(Contour)

            # upper left corner, lower right corner, red, thickness
            cv2.rectangle(image,(intX, intY), (intX+intW,intY+intH), (0, 0, 255),1)

            ROI = Thresh[intY:intY+intH, intX:intX+intW]
            ROIResized = cv2.resize(ROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            cv2.imshow("ROI", ROI)
            cv2.imshow("ROIResized", ROIResized)
            cv2.imshow("training_image.png", image)

            intChar = cv2.waitKey(0)

            if intChar == 27:
                sys.exit()
            elif intChar in intValidChars:

                Classifications.append(intChar)

                FlattenedImage = ROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                FlattenedImages = np.append(FlattenedImages, FlattenedImage, 0)                    # add current flattened impage numpy array to list of flattened image numpy arrays

    Classifications = np.array(Classifications, np.float32)

    Classifications = Classifications.reshape((Classifications.size, 1))

    print ("\n\ntraining complete !!\n")

    np.savetxt("classifications.txt", Classifications)
    np.savetxt("flattened_images.txt", FlattenedImages)

    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    main()
# end if




