import cv2
import scipy.io as sio
import numpy as np
from utility import showimg, draw_convex_py, edge_detect, find_contours, normalize_depth, draw_lf
from Lseg_to_Lfeat_py import create_linefeatures
from lineseg import lineseg
from merge_lines_v3 import merge_lines
from LabelLineCurveFeature_py import classify_curves
from test_convexity import test_convexity

if __name__ == '__main__':
    thresh_m = 10
    label_thresh = 11

    img = cv2.imread('learn0.png', -1)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', normalize_depth(img))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    data = sio.loadmat('input_LTLF_1.mat')
    ListSegLineC = data['ListSegLineC']
    ListEdgeC = data['ListEdgeC']

    data2 = sio.loadmat('LLCF_1.mat')
    Line_newCx = data2['Line_newCx']

    edges = edge_detect(img)
    cntrs = np.asarray(find_contours(edges))

    # test_convexity(cntrs, img)

    # Create line segments from the contours
    seglist = lineseg(cntrs, tol=2)
    # print(seglist)
    # print(cntrs)
    # print(ListSegLineC.shape)
    # print(ListEdgeC[0][0][:, 1])
    # print(cntrs[0][:, 0][:, 1])
    imgsize = (img.shape[1], img.shape[0])
    # print(imgsize)
    LineFeature, ListPoint = create_linefeatures(seglist, cntrs, imgsize)
    Line_new, ListPoint_new, line_merged = merge_lines(LineFeature, ListPoint, thresh_m, img.shape)
    # print(ListPoint_new)

    draw_lf(Line_new, img)

    line_newC = classify_curves(img, Line_new, ListPoint_new, label_thresh)

    # print(LineFeature)
    # print(ListPoint)

    draw_convex_py(line_newC, img)