import cv2
import scipy.io as sio
import numpy as np
from utility import showimg, draw_convex_py, edge_detect, find_contours, normalize_depth, draw_lf, draw_lp
from Lseg_to_Lfeat_py import create_linefeatures
from lineseg import lineseg
# from merge_lines_v3 import merge_lines
from merge_lines_py import merge_lines
from LabelLineCurveFeature_py import classify_curves
from drawedgelist import drawedgelist
from test_convexity import test_convexity


if __name__ == '__main__':
    thresh_m = 10
    label_thresh = 11

    img = cv2.imread('learn0.png', -1)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', normalize_depth(img, colormap=True))
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
    # drawedgelist(seglist, rowscols=[])
    # print(seglist)
    # print(cntrs)
    # print(ListSegLineC.shape)
    # print(ListEdgeC[0][0][:, 1])
    # print(cntrs[0][:, 0][:, 1])

    imgsize = (img.shape[1], img.shape[0])
    # imgsize = img.shape

    print(imgsize)
    LineFeature, ListPoint = create_linefeatures(seglist, cntrs, img, imgsize)
    print(LineFeature.shape)
    # print(np.ravel_multi_index((1, 1), imgsize, order='F'))
    Line_new, ListPoint_new, line_merged = merge_lines(LineFeature, ListPoint, thresh_m, imgsize)
    # Line_new, ListPoint_new, line_merged = merge_lines(LineFeature, ListPoint, thresh_m, img.shape)
    # print(line_merged)
    # draw_lf(LineFeature, img)
    # draw_lp(ListPoint, img, imgsize)
    draw_lf(Line_new, img)

    print(Line_new[0][8:10])
    print(ListPoint_new[0])
    line_newC = classify_curves(img, Line_new, ListPoint_new, label_thresh)

    draw_convex_py(line_newC, img)
