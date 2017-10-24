import cv2 as cv2
import numpy as np
import util as util
from scipy import stats

def grad_dir(img):
    # compute x and y derivatives
    # OpenCV's Sobel operator gives better results than numpy gradient
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)

    # calculate gradient direction angles
    # phase needs 64-bit input
    angle = cv2.phase(sobelx, sobely)

    # truncates number
    gradir = np.fix(180 + angle)

    return gradir


def curve_discont(depth_im):
    ###NEEDS IMPROVEMENT, NOT THAT GREAT ATM########
    # Gradient of depth img
    graddir = grad_dir(depth_im)

    # Threshold image to get it in the RGB color space
    dimg1 = (((graddir - graddir.min()) / (graddir.max() - graddir.min())) * 255.9).astype(np.uint8)

    # Further remove noise while keeping edges sharp
    blur = cv2.bilateralFilter(dimg1, 9, 25, 25)
    blur2 = cv2.bilateralFilter(blur, 9, 25, 25)

    # Eliminate salt-and-pepper noise
    median = cv2.medianBlur(blur2, 7)

    dimg1 = util.auto_canny(median)
    skel1 = util.morpho(dimg1)

    # util.showimg(util.create_img(skel1), "Morphology + canny on depth image")


    ######CAN'T FIND USE FOR CNT1, what is the point of finding contours here?########
    #cnt1 = util.find_contours(util.create_img(skel1), cv2.RETR_EXTERNAL)

    return skel1