# Robot Grasping Algorithm

# Importing the necessary packages:
import cv2
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from skimage import morphology

from utility import showimg, normalize_depth, edge_detect, find_contours
from drawlinefeature import DrawLineFeature,drawconvex
from lineseg import lineseg
from drawedgelist import drawedgelist
from Lseg_to_Lfeat_v2 import Lseg_to_Lfeat_v2
from LabelLineCurveFeature_v2 import LabelLineCurveFeature_v2
from merge_lines_v2 import merge_lines
from LabelLineCurveFeature import classify_curves


if __name__ == '__main__':
    # second argument is a flag which specifies the way
    # an image should be read. -1 loads image unchanged with alpha channel
    depthimg = cv2.imread('img/learn15.png', -1)
    colorimg = cv2.imread('img/clearn17.png', 0)

    # showimg(normalize_depth(depthimg, colormap=True), 'depth')

    id = depthimg[100:, 100:480]  ## zc crop the region of interest

    siz = id.shape  ## image size of the region of interest
    print(depthimg.shape)
    thresh_m = 10
    label_thresh = 11
    # edges = edge_detect(depthimg, colorimg)
    edges = edge_detect(id)  # zc

    showimg(edges, "Canny of depth image + discontinuity")
    # showimg(cntr1)
    # showimg(cntr2)

    cntrs = np.asarray(find_contours(edges))

    seglist = lineseg(cntrs, tol=2)
    drawedgelist(seglist, rowscols=[])

    # Get line features for later processing
    [linefeature, listpoint] = Lseg_to_Lfeat_v2(seglist, cntrs, siz)

    # Data to test merge_lines
    data1 = sio.loadmat('mergeline_input1_LineFeatureC.mat')
    LineFeatureC = data1['LineFeatureC']
    data2 = sio.loadmat('mergeline_input2_ListPointC.mat')
    ListPointC = data2['ListPointC']
    data3 = sio.loadmat('mergeline_output1_Line_newC.mat')
    Line_new = list(data3['Line_newC'])
    data4 = sio.loadmat('mergeline_output2_ListPoint_newC.mat')
    ListPoint_newC = data4['ListPoint_newC']
    data5 = sio.loadmat('mergeline_output3_Line_merged_nC.mat')
    Line_merged_nC = data5['Line_merged_nC']
    # print('LineFeatureC', LineFeatureC)
    # print('ListPointC', ListPointC.shape)
    # print('Line_merged_nC', Line_merged_nC)
    [line_new, listpoint_new, line_merged] = merge_lines(linefeature, listpoint, thresh_m, siz)
    # [line_new, listpoint_new, line_merged] = merge_lines(linefeature, listpoint, thresh_m, siz)

    # line_new = LabelLineCurveFeature_v2(depthimg, line_new, listpoint_new, label_thresh)
    line_new = classify_curves(depthimg, line_new, listpoint_new, label_thresh)
    # line_new = LabelLineCurveFeature_v2(depthimg, line_new, listpoint_new, label_thresh)
    DrawLineFeature(linefeature, siz, 'line features')
    drawconvex(line_new, siz, 'convex')

    # TO-DO
    # - Check LabelLineCurveFeature_v2 with python input
    # - Write a function to make a window mask
    # - (Section 4) Take non-zero curve features and sort by angle (index 7 in MATLAB)

    # Long-term, to make the program easier to read and use
    # *- Create a Line object with properties: start, end, object/background, curvature/discontinuity
    #   distance from another line, check if line is overlapping
