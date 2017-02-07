import cv2
import scipy.io as sio
import numpy as np
from merge_lines_v3 import merge_lines
from utility import showimg
from Lseg_to_Lfeat_v3 import create_linefeatures


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
    img = cv2.imread('learn15.png', -1)

    # showimg(normalize_depth(depthimg, colormap=True), 'depth')

    id = img[100:, 100:480]  ## zc crop the region of interest
    # print("id:", id.shape)

    siz = img.shape  ## image size of the region of interest
    # print(siz)
    thresh_m = 10

    data = sio.loadmat('in_out_ML_2.mat')
    # inputs
    LineFeatureC = data['LineFeatureC']
    ListPointC = data['ListPointC']
    # LineFeature = data['LineFeature']
    # ListPoint = data['ListPoint']
    # output
    Line_newC = data['Line_newC']
    ListPoint_newC = data['ListPoint_newC']
    Line_merged_nC = data['Line_merged_nC']

    # data = sio.loadmat('input_LTLF_1.mat')
    # data2 = sio.loadmat('output_LTLF_1.mat')

    # data = sio.loadmat('input_LTLF_1.mat')
    # data2 = sio.loadmat('output_LTLF_1.mat')
    # data3 = sio.loadmat('out_ML_1.mat')
    #
    # ListSegLineC = data['ListSegLineC']
    # ListEdgeC = data['ListEdgeC']
    #
    # LineFeatureC = data2['LineFeatureC']
    # ListPointC = data2['ListPointC']
    #
    # Line_newC = data3['Line_newC']
    # ListPoint_newC = data3['ListPoint_newC']
    # Line_merged_nC = data3['Line_merged_nC']
    #
    # LineFeature, ListPoint = create_linefeatures(ListSegLineC, ListEdgeC, img.shape)

    # print(ListPointC[550])
    line_new, listpoint_new, line_merged = merge_lines(LineFeatureC, ListPointC, thresh_m, siz)
    print("Line_newC shape:", Line_newC.shape)
    print("ListPoint_newC shape:", ListPoint_newC.shape)
    print("Line_merged_nC shape:", Line_merged_nC.shape)
    print("======================================")

    print("Line_new shape:", line_new.shape)
    print("Listpoint_new shape:", len(listpoint_new))
    print("line_merged shape:", line_merged.shape)

    # print(listpoint_new)
    # print("line merged\n")
    # print(*line_merged, sep='\n')

    matches = 0
    for i in range(len(listpoint_new)):
        if np.array_equiv(listpoint_new[i], ListPoint_newC[i][0]): # listpoint_new[i] == ListPoint_newC[i][0]:
            matches += 1
        # else:
        #     print(i, '.\n', listpoint_new[i], sep='')
        #     print(ListPoint_newC[i][0])
    # print('matches', matches)
    # print(Line_merged_nC.shape, len(line_merged))
    # length = len(line_merged)
    # for i in range(0, length-1):
    #     if(i > Line_merged_nC.shape[0]):
    #         print(" ", line_merged[i], "\n")
    #     else:
    #         print(Line_merged_nC[i], " ", line_merged[i])

    # print("line_new", line_new)
