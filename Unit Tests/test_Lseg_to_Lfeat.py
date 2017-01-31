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

    create_linefeatures(ListSegLineC, ListEdgeC, img.shape)

