import cv2
import scipy.io as sio
import numpy as np
from merge_lines_v3 import merge_lines
from utility import showimg


def normalize_depth(depthimg, colormap=False):
    # Normalize depth image
    min, max, minloc, maxloc = cv2.minMaxLoc(depthimg)
    adjmap = np.zeros_like(depthimg)
    dst = cv2.convertScaleAbs(depthimg, adjmap, 255 / (max - min), -min)
    if colormap == True:
        return cv2.applyColorMap(dst, cv2.COLORMAP_JET)
    else:
        return dst


if __name__ == '__main__':
    depthimg = cv2.imread('learn15.png', -1)

    # showimg(normalize_depth(depthimg, colormap=True), 'depth')

    id = depthimg[100:, 100:480]  ## zc crop the region of interest
    print("id:", id.shape)

    siz = depthimg.shape  ## image size of the region of interest
    print(siz)
    thresh_m = 10

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
    # print('Line_new', Line_new)
    # print('ListPoint_newC', ListPoint_newC)
    # print('Line_merged_nC', Line_merged_nC)
    [line_new, listpoint_new, line_merged] = merge_lines(LineFeatureC, ListPointC, thresh_m, siz)
    print("line merged", line_merged)
    # print(Line_merged_nC.shape, len(line_merged))
    # length = len(line_merged)
    # for i in range(0, length-1):
    #     if(i > Line_merged_nC.shape[0]):
    #         print(" ", line_merged[i], "\n")
    #     else:
    #         print(Line_merged_nC[i], " ", line_merged[i])

    # print("line_new", line_new)
