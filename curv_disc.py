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
    #cv2.imshow("bruh-1", depth_im)
    graddir = grad_dir(depth_im)
    #print(np.amin(graddir), "min", np.amax(graddir), "max")
    # Threshold image to get it in the RGB color space
    dimg1 = (((graddir - graddir.min()) / (graddir.max() - graddir.min())) * 255.9).astype(np.uint8)
    #print(dimg1, "dimg1")
    ####For fixing holes in just curve image#####
    """backgroundVal = stats.mode(dimg1, axis=None)[0]
                print(backgroundVal, "backgroundVal")
                dimg1 = util.fixHoles(depth_im, dimg1, backgroundVal)"""

    cv2.imshow("bruh", dimg1)
    # Eliminate salt-and-pepper noise
    median = cv2.medianBlur(dimg1, 13)
    cv2.imshow("bruh2", median)
    # Further remove noise while keeping edges sharp
    blur = cv2.bilateralFilter(median, 9, 25, 25)
    cv2.imshow("bruh3", blur)
    dimg1 = util.auto_canny(blur)
    skel1 = util.morpho(dimg1)
    
    #util.showimg(util.create_img(skel1), "Morphology + canny on depth image")


    ######CAN'T FIND USE FOR CNT1, what is the point of finding contours here?########
    cnt1 = util.find_contours(util.create_img(skel1), cv2.RETR_EXTERNAL)

    return skel1, cnt1