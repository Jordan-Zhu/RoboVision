import numpy as np
import cv2
from maskedMedianFilter import masked_median_filter
from algo import showimg


def zeroElimMedianHoleFill(im):
    mask = np.where(np.array(im) == 0)
    r = masked_median_filter(im, mask)
    has_holes = ~np.all(np.all(r))
    while has_holes:
        r = masked_median_filter(r, r == 0)
        has_holes = ~np.all(np.all(r))
    return r

if __name__ == '__main__':
    depthimg = cv2.imread('img/learn17.png', -1)
    h = zeroElimMedianHoleFill(depthimg)
    showimg(h)