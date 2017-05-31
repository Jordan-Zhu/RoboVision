import cv2
import numpy as np
import math
from ZeroElimMedianFilter_v1 import zeroElimMedianFilter
from algo import showimg

def zeroElimMedianHoleFill(im):
    n = 1
    r = zeroElimMedianFilter(im)
    hasHoles = ~all(r.all())

    while hasHoles:
        r = zeroElimMedianFilter(r)
        hasHoles = ~all(r.all())
        n += 1

    return r, n

if __name__ == '__main__':
    depthimg = cv2.imread('img/learn17.png', -1)
    h = zeroElimMedianHoleFill(depthimg)
    showimg(h)