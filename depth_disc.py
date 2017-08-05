import cv2 as cv2
import numpy as np
import util as util
import settings


def depth_discont(depth_im):
    # Depth discontinuity
    # depthimg = util.normalize_depth(depth_im)
    # dimg2 = clahe(depthimg, iter=2)
    #depth_im = util.fixHoles2(depth_im)
    dimg2 = util.auto_canny(depth_im)
    skel2 = util.morpho(dimg2)
    if settings.dev_mode is True:
        cv2.imshow("discontinuity", dimg2)
    #util.showimg(dimg2, "Discontinuity")
    #cnt2 = util.find_contours(util.create_img(skel2), cv2.RETR_EXTERNAL)

    return skel2