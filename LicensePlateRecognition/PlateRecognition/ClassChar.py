
import cv2
import numpy as np
import math

class Char:

    # constructor
    def __init__(Char, contour):
        Char.contour = contour

        Char.boundingRect = cv2.boundingRect(Char.contour)

        [X, Y, Width, Height] = Char.boundingRect

        Char.BoundingRectX = X
        Char.BoundingRectY = Y
        Char.BoundingRectWidth = Width
        Char.BoundingRectHeight = Height

        Char.DiagonalSize = math.sqrt((Char.BoundingRectWidth ** 2) + (Char.BoundingRectHeight ** 2))

        Char.AspectRatio = float(Char.BoundingRectWidth) / float(Char.BoundingRectHeight)

        Char.BoundingRectArea = Char.BoundingRectWidth * Char.BoundingRectHeight

        Char.CenterX = (Char.BoundingRectX + Char.BoundingRectX + Char.BoundingRectWidth) / 2
        Char.CenterY = (Char.BoundingRectY + Char.BoundingRectY + Char.BoundingRectHeight) / 2
