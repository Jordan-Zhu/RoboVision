<<<<<<< Updated upstream
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

def normalize_depth(depthimg, colormap=False):
    # Normalize depth image
    min, max, minloc, maxloc = cv2.minMaxLoc(depthimg)
    adjmap = np.zeros_like(depthimg)
    dst = cv2.convertScaleAbs(depthimg, adjmap, 255 / (max - min), -min)
    if colormap == True:
        return cv2.applyColorMap(dst, cv2.COLORMAP_JET)
    else:
        return dst

if __name__ == '__main__':
    img = cv2.imread('img/learn17.png', -1)
    # img = np.nonzero(img)
    # h = zeroElimMedianHoleFill(depthimg)
    showimg(normalize_depth(img))
    median = cv2.medianBlur(img, 5)
    showimg(normalize_depth(median))
=======
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
>>>>>>> Stashed changes
