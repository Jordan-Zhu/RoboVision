import cv2 as cv2
import numpy as np
import random as rand
from skimage import morphology


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def normalize_depth(depthimg, colormap=False):
    # Normalize depth image to range 0-255.
    min, max, minloc, maxloc = cv2.minMaxLoc(depthimg)
    adjmap = np.zeros_like(depthimg)
    dst = cv2.convertScaleAbs(depthimg, adjmap, 255 / (max - min), -min)
    if colormap == True:
        return cv2.applyColorMap(dst, cv2.COLORMAP_JET)
    else:
        return dst


def morpho(img):
    kernel = np.ones((7, 7), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    skel = morphology.skeletonize(dilation > 0)

    return skel


def showimg(img, im_name='image', write=False, imagename='img.png'):
        cv2.namedWindow(im_name, cv2.WINDOW_NORMAL)
        cv2.imshow(im_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if write:
            cv2.imwrite("../../images/%s", imagename, img)


def find_contours(im, mode=cv2.RETR_CCOMP):
    # im = cv2.imread('circle.png')
    # error: (-215) scn == 3 || scn == 4 in function cv::ipp_cvtColor
    # imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    height = im.shape[0]
    width = im.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8)
    img = normalize_depth(im, colormap=True)
    if mode == cv2.RETR_CCOMP:
        im2, contours, hierarchy = cv2.findContours(im, mode, cv2.CHAIN_APPROX_NONE)
        newcontours = []
        for i in range(len(contours)):
            if hierarchy[0][i, 2] < 0:
                # color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
                # cv2.drawContours(blank_image, contours, i, color, 1, 8)
                newcontours.append(contours[i])

        # Display contours
        # draw_contours(blank_image, contours)
        # cv2.imshow("CONTOURS", blank_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cntrs = np.array(newcontours)
        # print(cntrs)
        sqz_contours(cntrs)

        return cntrs
    else:
        im2, contours, hierarchy = cv2.findContours(im, mode, cv2.CHAIN_APPROX_NONE)
        # draw_contours(blank_image, contours)
    # cv2.RETR_EXTERNAL cv2.RETR_CCOMP

    cntrs = np.array(contours)
    sqz_contours(cntrs)

    return cntrs


def swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]]


def squeeze_ndarr(arr):
    temp = []
    for i in range(arr.shape[0]):
        temp.append(np.squeeze(arr[i]))
    np.copyto(arr, np.array(temp))


# Squeezes the array and swaps the columns to match Numpy's col, row ordering
def sqz_contours(contours):
    squeeze_ndarr(contours)
    # ADVANCED SLICING
    for i in range(contours.shape[0]):
        swap_cols(contours[i], 0, 1)


def draw_contours(im, contours):
    height = im.shape[0]
    width = im.shape[1]
    overlay = np.zeros((height, width, 3), np.uint8)
    output = np.zeros((height, width, 3), np.uint8)
    alpha = 0.5

    # cv2.putText(overlay, "ROI Poly: alpha={}".format(alpha), (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    for i in range(len(contours)):
        color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
        cv2.drawContours(im, contours, i, color, 1, 8)
        # cv2.imshow("contours", im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # cv2.addWeighted(overlay, alpha, output, 1 - alpha,
    #                 0, output)
    cv2.imshow("contours", im)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


def draw_lf(line_feature, img):
    # blank_image = normalize_depth(img, colormap=True)
    blank_image = np.zeros_like(img)

    # print(line_feature[0])
    for i, e in enumerate(line_feature):
        x1 = int(e[1])
        y1 = int(e[0])
        x2 = int(e[3])
        y2 = int(e[2])
        color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
        cv2.line(blank_image, (x1, y1), (x2, y2), color, 2)
        # cv2.namedWindow('Convex lines', cv2.WINDOW_NORMAL)
        # cv2.imshow('Convex lines', blank_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    cv2.namedWindow('Line features', cv2.WINDOW_NORMAL)
    cv2.imshow('Line features', blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_listpair(list_pair, line_feature, img):
    blank_image = normalize_depth(img, colormap=True)

    for i, e in enumerate(list_pair):
        color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
        for j, e in enumerate(e):
            line = line_feature[np.where(line_feature[:, 7] == e)[0]][0]
            x1 = int(line[1])
            y1 = int(line[0])
            x2 = int(line[3])
            y2 = int(line[2])
            cv2.line(blank_image, (x1, y1), (x2, y2), color, 2)

    cv2.namedWindow('Line features', cv2.WINDOW_NORMAL)
    cv2.imshow('Line features', blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def swap_indices(arr):
    res = []
    for i, e in enumerate(arr):
        res.append([arr[i][1], arr[i][0]])
    return np.array(res)


def create_img(mat):
    blank_image = np.zeros((mat.shape[0], mat.shape[1], 3), np.uint8)
    mask = np.array(mat * 255, dtype=np.uint8)
    masked = np.ma.masked_where(mask <= 0, mask)

    return mask