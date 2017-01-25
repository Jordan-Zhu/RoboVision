import cv2
import scipy.io as sio
import numpy as np

from merge_lines_v3 import merge_lines
from LabelLineCurveFeature import classify_curves
from utility import normalize_depth


def roipoly(src, line, poly):
    mask = np.zeros_like(src)
    dst = np.zeros_like(src)
    cv2.rectangle(mask, (poly[0][1], poly[0][0]), (poly[3][1], poly[3][0]), (255, 255, 255), cv2.FILLED)
    cv2.bitwise_and(src, dst, mask=mask)
    cv2.imshow("image", dst)


if __name__ == '__main__':
    depthimg = cv2.imread('learn15.png', -1)
    src = normalize_depth(depthimg, colormap=True)
    poly = [[100, 100], [0, 0], [0,0], [200, 200]]

    siz = depthimg.shape
    thresh_m = 10
    label_thresh = 11

    mask = np.zeros_like(src)
    dst = np.zeros_like(src)
    cv2.rectangle(mask, (poly[0][1], poly[0][0]), (poly[3][1], poly[3][0]), (255, 255, 255), cv2.FILLED)
    res = cv2.bitwise_and(src, dst, mask=mask)
    # res = np.copyto(dst, src, where=)
    cv2.imshow("image", mask)
    cv2.waitKey(0)

    data = sio.loadmat('LabelLineCurveFeature_v2.mat')
    # data2 = sio.loadmat('Parameter.mat')
    #
    # # inputs
    Line_newC = data['Line_newC']
    ListPoint_newC = data['ListPoint_newC']
    #
    # Parameter = data2['P']

    # [line_new, listpoint_new, line_merged] = merge_lines(Line_newC, ListPoint_newC, thresh_m, siz)
    # line_new = classify_curves(depthimg, Line_newC, ListPoint_newC, label_thresh)
