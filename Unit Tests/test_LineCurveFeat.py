import cv2
import scipy.io as sio

from merge_lines_v3 import merge_lines
from LabelLineCurveFeature import classify_curves
from utility import normalize_depth


if __name__ == '__main__':
    depthimg = cv2.imread('learn15.png', -1)

    siz = depthimg.shape
    thresh_m = 10
    label_thresh = 11

    data = sio.loadmat('LabelLineCurveFeature_v2.mat')
    # data2 = sio.loadmat('Parameter.mat')
    #
    # # inputs
    Line_newC = data['Line_newC']
    ListPoint_newC = data['ListPoint_newC']
    #
    # Parameter = data2['P']

    # [line_new, listpoint_new, line_merged] = merge_lines(Line_newC, ListPoint_newC, thresh_m, siz)
    line_new = classify_curves(depthimg, Line_newC, ListPoint_newC, label_thresh)
