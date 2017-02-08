import cv2
import scipy.io as sio
import numpy as np
from merge_lines_v3 import merge_lines
from utility import showimg
from Lseg_to_Lfeat_v3 import create_linefeatures
from LabelLineCurveFeature import classify_curves


if __name__ == '__main__':
    thresh_m = 10
    label_thresh = 11

    img = cv2.imread('learn0.png', -1)

    data = sio.loadmat('input_LTLF_1.mat')
    ListSegLineC = data['ListSegLineC']
    ListEdgeC = data['ListEdgeC']

    # SEGMENT AND LABEL THE CURVATURE LINES AS EITHER CONVEX / CONCAVE
    LineFeature, ListPoint = create_linefeatures(ListSegLineC, ListEdgeC, img.shape)
    Line_new, ListPoint_new, line_merged = merge_lines(LineFeature, ListPoint, thresh_m, img.shape)
    line_newC = classify_curves(img, Line_new, ListPoint_new, label_thresh)

    print(line_newC[:, 10])




