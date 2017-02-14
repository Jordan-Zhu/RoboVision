import cv2
import scipy.io as sio
import numpy as np
from merge_lines_v3 import merge_lines
from utility import showimg, draw_convex, edge_detect, find_contours
from Lseg_to_Lfeat_v3 import create_linefeatures
from LabelLineCurveFeature import classify_curves
from lineseg import lineseg


if __name__ == '__main__':
    thresh_m = 10
    label_thresh = 11

    img = cv2.imread('learn0.png', -1)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    Id_o = sio.loadmat('Id.mat')
    Id = Id_o['Id']
    Id = Id.astype(np.uint8).copy()

    data = sio.loadmat('input_LTLF_1.mat')
    ListSegLineC = data['ListSegLineC']
    ListEdgeC = data['ListEdgeC']

    data2 = sio.loadmat('LLCF_1.mat')
    Line_newCx = data2['Line_newCx']

    # edges = edge_detect(img)
    # cntrs = np.asarray(find_contours(edges))
    #
    # # Create line segments from the contours
    # seglist = lineseg(cntrs, tol=2)
    # print(seglist[0])

    # SEGMENT AND LABEL THE CURVATURE LINES AS EITHER CONVEX / CONCAVE
    LineFeature, ListPoint = create_linefeatures(ListSegLineC, ListEdgeC, img.shape)
    # LineFeature, ListPoint = create_linefeatures(seglist, edges, img.shape)
    Line_new, ListPoint_new, line_merged = merge_lines(LineFeature, ListPoint, thresh_m, img.shape)
    line_newC = classify_curves(img, Line_new, ListPoint_new, label_thresh)

    print(line_newC.shape)
    print(line_newC[:, 10])

    sum = 0
    for i in range(len(line_newC)):
        if line_newC[i, 10] == Line_newCx[i, 10]:
            # print(i, ".", int(line_newC[i, 10]), " == ", int(Line_newCx[i, 10]), "\n")
            sum += 1
        else:
            print(i, ".", int(line_newC[i, 10]), " == ", int(Line_newCx[i, 10]), "\n")
    print('Total lines:', len(line_newC), ' Total correct lines:', sum)

    draw_convex(line_newC, img)




