import cv2
import scipy.io as sio
import numpy as np
import util as util

from edge_detect import edge_detect
from lineseg import lineseg
from drawedgelist import drawedgelist

import Lseg_to_Lfeat_v4 as Lseg_to_Lfeat_v4
import merge_lines_v4 as merge_lines_v4
import LabelLineCurveFeature_v4 as LabelLineCurveFeature_v4
import LabelLineFeature_v1 as LabelLineFeature_v1
from line_match import line_match


if __name__ == '__main__':
    
    ##These methods are for the picture resizing

    mouseX = []
    mouseY = []
    numC = 0

    def choosePoints(event,x,y,flags,param):
        global mouseX,mouseY, numC
        

        if event == 4:
            #cv2.circle(img,(x,y),100,(255,0,0),-1)
            print(x,y)
            numC += 1
            print(numC, "clicked")
            mouseX.append(x)
            mouseY.append(y)
            print(mouseX, mouseY, "bruh")

    imgC = cv2.imread('img/clearn0.png', -1)

    cv2.imshow('image',imgC)
    cv2.setMouseCallback('image', choosePoints)
    while(numC != 2):
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            print(mouseX, mouseY, "printing mousey")
            break

    print(mouseX, mouseY, "printing mousey")
    cv2.destroyAllWindows()

    # Read in depth image, -1 specifies w/ alpha channel.
    img = cv2.imread('img/learn0.png', -1)
    #input_var = input("Enter x1, x2, y1, y2 with spaces in between (put 0 0 0 0 for no change) ")
    #dimensions = list(map(int, input_var.split()))
    #print(dimensions, "dimensions")
    #if(dimensions.count(0) != 4):
    #    img = img[dimensions[0]:dimensions[1], dimensions[2]:dimensions[3]]
    #cv2.namedWindow('imageResize')
    
    #cv2.destroyAllWindows()

    img = img[mouseY[0]:mouseY[1], mouseX[0]:mouseX[1]]
    imgC = imgC[mouseY[0]:mouseY[1], mouseX[0]:mouseX[1]]
    cv2.imshow('cropped', imgC)
    cv2.waitKey(0)

    im_size = img.shape
    height = img.shape[0]
    width = img.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8)

    P = sio.loadmat('Parameter.mat')
    param = P['P']

    # ******* SECTION 1 *******
    # FIND DEPTH / CURVATURE DISCONTINUITIES.
    curve_disc, curve_con, depth_disc, depth_con, dst = edge_detect(img)

    # Remove extra dimensions from data
    res = lineseg(dst, tol=2)
    seglist = []
    for i in range(res.shape[0]):
        # print('shape', res[i].shape)
        if res[i].shape[0] > 2:
            # print(res[i])
            # print(res[i][0])
            seglist.append(np.concatenate((res[i], [res[i][0]])))
        else:
            seglist.append(res[i])

    seglist = np.array(seglist)

    drawedgelist(seglist)

    # ******* SECTION 2 *******
    # SEGMENT AND LArrrBEL THE CURVATURE LINES (CONVEX/CONCAVE).
    LineFeature, ListPoint = Lseg_to_Lfeat_v4.create_linefeatures(seglist, dst, im_size)
    Line_new, ListPoint_new, line_merged = merge_lines_v4.merge_lines(LineFeature, ListPoint, 10, im_size)

    util.draw_lf(Line_new, blank_image)

    line_newC = LabelLineCurveFeature_v4.classify_curves(img, Line_new, ListPoint_new, 11)

    # # Drop the angle 11th column
    # line_newC = np.delete(line_newC, 10, axis=1)
    # line_new_new = LabelLineFeature_v1.label_line_features(img, edges, line_newC, param)
    # print('Line_new:', line_new_new.shape)
    #
    # # ******* SECTION 4 *******
    # # SELECT THE DESIRED LINES FROM THE LIST
    #
    # # Keep the lines that are curvature / discontinuities
    # relevant_lines = np.where(line_new_new[:, 10] != 0)[0]
    # line_interesting = line_new_new[relevant_lines]
    # # Sort lines in ascending order based on angle
    # line_interesting = line_interesting[line_interesting[:, 6].argsort()]
    #
    # print('Line interesting:', line_interesting.shape)
    # util.draw_lfeat(line_interesting, img)
    #
    # # Match the lines into pairs
    # list_pair = line_match(line_interesting, param)
    # print('List pair:', list_pair)
    # util.draw_listpair(list_pair, line_interesting, img)