import cv2
import numpy as np
import scipy.io as sio

from Lseg_to_Lfeat_v3 import create_linefeatures
from Lseg_to_Lfeat_v2 import Lseg_to_Lfeat_v2


if __name__ == '__main__':
    img = cv2.imread('learn0.png', -1)

    data = sio.loadmat('input_LTLF_1.mat')
    data2 = sio.loadmat('output_LTLF_1.mat')

    ListSegLineC = data['ListSegLineC']
    ListEdgeC = data['ListEdgeC']

    LineFeatureC = data2['LineFeatureC']
    ListPointC = data2['ListPointC']

    LineFeature, LPP = create_linefeatures(ListSegLineC, ListEdgeC, img.shape)

    lf_len = len(LineFeature)
    lf_match = 0
    lp_match = 0
    for i in range(lf_len):
        if all(LineFeature[i]) == all(LineFeatureC[i]):
            lf_match += 1
        else:
            print(i, ".")
            print(LineFeature[i])
            print(LineFeatureC[i])
        if all(LPP[i][0]) == all(ListPointC[i][0]):
            lp_match += 1
        else:
            print(LPP[i][0])
            print(ListPointC[i][0])
    # print(LPP[0][0])
    print("--------------------")
    print(LineFeature[1])
    print(LineFeatureC[1])
    print(LineFeature[2])
    print(LineFeatureC[2])
    print("Totals:\nLineFeature:", lf_len, "ListPoint:", len(ListPointC))
    print("LineFeature matching lines:", lf_match)
    print("ListPoint matches:", lp_match)


