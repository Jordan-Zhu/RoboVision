import cv2 as cv2
import numpy as np
import util as util


# Contrast Limited Adaptive Histogram Equalization
# Improves the contrast of our image.
def clahe(img, iter=1):
    for i in range(0, iter):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    return img


def depth_discont(depth_im):
    # Depth discontinuity
    depthimg = util.normalize_depth(depth_im)
    dimg2 = clahe(depthimg, iter=2)
    dimg2 = util.auto_canny(dimg2)
    skel2 = util.morpho(dimg2)
    # util.showimg(dimg2, "Depth discontinuity w/ tone balancing")
    cnt2 = util.find_contours(util.create_img(skel2), cv2.RETR_EXTERNAL)

    return skel2, cnt2