import numpy as np
import cv2

def translate(image, x, y):
    M = np.float32([[1, 0, x], [1, 0, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def rotate(image, degrees, scale):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, degrees, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def resize(image, width = None, height = None):
    (h, w) = image.shape[:2]

    if width == None and height == None:
        return image
    elif width == None:
        r = height / h
        dim = (int(w*r), int(height))
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        return resized
    elif height == None:
        r = width / w
        dim = (int(width), int(h*r))
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        return resized
    
def contours(cnts):
    return cnts[0]