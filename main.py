import cv2

import scipy.io as sio

import numpy as np

import util as util



from edge_detect import edge_detect

from lineseg import lineseg

from drawedgelist import drawedgelist


import classify_curves as cc
import label_curves as lc
import Line_feat_contours as lfc
import Lseg_to_Lfeat_v4 as Lseg_to_Lfeat_v4

import merge_lines_v4 as merge_lines_v4

import LabelLineCurveFeature_v4 as LabelLineCurveFeature_v4

import LabelLineFeature_v1 as LabelLineFeature_v1

from line_match import line_match

np.set_printoptions(threshold=np.nan)

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt



if __name__ == '__main__':

    for numImg in [3]:

        ##These methods are for the picture resizing

        mouseX = []

        mouseY = []

        numC = 0

    

        ###Event 4 means that the right key was clicked

        ###This saves the points that are clicked on the image

        def choosePoints(event,x,y,flags,param):

            global mouseX,mouseY, numC

            

    

            if event == 4:

                #cv2.circle(img,(x,y),100,(255,0,0),-1)

                numC += 1

                mouseX.append(x)

                mouseY.append(y)

        

        #Opens up the color image for user to click on

        imgC = cv2.imread('img/clearn%d.png' %numImg, -1)

        cv2.imshow('image',imgC)

        cv2.setMouseCallback('image', choosePoints)

        

        #checks and makes sure 2 points were clicked

        #if 2 points were clicked it exits the loop

        while(numC != 2):

            key = cv2.waitKey(1) & 0xFF

            

            """if key == ord("r"):

                print(mouseX, mouseY, "printing mousey")

                break"""

        

        

        #Closes color image once user clicks twice

        cv2.destroyAllWindows()

    

        # Read in depth image, -1 means w/ alpha channel.

        # This keeps in the holes with no depth data as just black.

        depth_im = 'img/learn%d.png'%numImg

        img = cv2.imread(depth_im, -1)

  

        #crops the depth image

        img = img[mouseY[0]:mouseY[1], mouseX[0]:mouseX[1]]
        final_im = util.normalize_depth(img, colormap=True)
        #img = util.fixHoles(img)

        #For convenience, to see what you cropped

        imgC = imgC[mouseY[0]:mouseY[1], mouseX[0]:mouseX[1]]

        cv2.imshow('cropped', imgC)

        

        #cv2.waitKey(0)

        im_size = img.shape

        height = img.shape[0]

        width = img.shape[1]

        blank_image = np.zeros((height, width, 3), np.uint8)

        

        P = sio.loadmat('Parameter.mat')

        param = P['P']



    

        # evenly increases the contrast of the entire image

        # ref: http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html

        def clahe(img, iter=1):

            for i in range(0, iter):

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

                img = clahe.apply(img)

            return img

    

    

        # Open a copy of the depth image

        # to change the contrast on the full-sized image

        img2 = cv2.imread(depth_im, -1)

        old_height = img2.shape[0]

        old_width = img2.shape[1]
        
        old_blank_image = np.zeros((old_height, old_width, 3), np.uint8)
        
        util.depthToPC(img2, old_blank_image, 320, 240, 300, mouseY[0], mouseX[0])


        img2 = util.normalize_depth(img2)

        img2 = clahe(img2, iter=2)
        

        # crops the image

        img2 = img2[mouseY[0]:mouseY[1], mouseX[0]:mouseX[1]]

        




    

    

        # ******* SECTION 1 *******

        # FIND DEPTH / CURVATURE DISCONTINUITIES.

        curve_disc, curve_con, depth_disc, depth_con, edgelist = edge_detect(img, img2, img, numImg)

    

        # Remove extra dimensions from data

        res = lineseg(edgelist, tol=2)

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
        print(edgelist.shape, "edgelist shape")
        # print(edgelist[0])
        print(seglist.shape, "seglist shape")

    

        drawedgelist(seglist, blank_image, numImg)

    

        # ******* SECTION 2 *******

        # SEGMENT AND LABEL THE CURVATURE LINES (CONVEX/CONCAVE).
        for j in range(seglist.shape[0]):
            # LineFeature, ListPoint = Lseg_to_Lfeat_v4.create_linefeatures(seglist, dst, im_size)
            LineFeature, ListPoint = lfc.create_linefeatures(seglist[j], j, edgelist, im_size)

            # print("angles\n", LineFeature[:, 6])

            Line_new, ListPoint_new, line_merged = merge_lines_v4.merge_lines(LineFeature, ListPoint, 20, im_size)
            print(line_merged, "merged")
            print("angles\n", Line_new[:, 6])
            # Line_new = LineFeature
            # ListPoint_new = ListPoint

            # print(Line_new.shape, "NEW line size")
            # print(ListPoint_new.shape, "NEW list point size")
            # blank_im = np.zeros((height, width, 3), np.uint8)
            # for i in range(ListPoint_new.shape[0]):
            #     y, x = np.unravel_index(ListPoint_new[i], im_size, order='F')
            #     x = np.squeeze(x)
            #     y = np.squeeze(y)
            #     print(x, "x")
            #     print(y, "y")
            #     for i, e in enumerate(Line_new):
            #         x1 = int(e[1])
            #         y1 = int(e[0])
            #         x2 = int(e[3])
            #         y2 = int(e[2])
            #         color = (0, 255, 0)
            #         cv2.line(blank_im, (x1, y1), (x2, y2), color, thickness=1)
            #         # blank_im2 = np.zeros((height, width, 3), np.uint8)
            #         # cv2.line(blank_im2, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
            #         cv2.imshow("Current Line", blank_im)
            #
            #         for i in range(x.shape[0]):
            #             x = int(e[1])
            #             y = int(e[0])
            #             color = (0, 0, 255)
            #             cv2.line(blank_im, (x, y), (x, y), color, thickness=1)
            #
            #         cv2.imshow("List Edges", blank_im)
            #         cv2.waitKey(0)
            # for i, e in enumerate(ListPoint_new):
            #     y1, x1 = np.unravel_index([lind1], im_size, order='F')
            #     x = int(e[1])
            #     y = int(e[0])
            #     x1 = int(Line_new[i][1])
            #     y1 = int(Line_new[i][0])
            #     x2 = int(Line_new[i][3])
            #     y2 = int(Line_new[i][2])
            #     color = (0, 255, 0)
            #     cv2.line(blank_im, (x, y), (x, y), color, thickness=1)
            #     print("x1", x1, "y1", y1, "x2", x2, "y2", y2)
            #     blank_im2 = np.zeros((height, width, 3), np.uint8)
            #     cv2.line(blank_im2, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
            #     cv2.imshow("Current Line", blank_im2)
            #     cv2.imshow("List Edges", blank_im)
            #     cv2.waitKey(0)


            util.draw_lf(Line_new, blank_image, numImg)



            # line_newC = LabelLineCurveFeature_v4.classify_curves(curve_disc, depth_disc, Line_new, ListPoint_new, 11)
            line_new = cc.classify_curves(curve_disc, depth_disc, Line_new, ListPoint_new, 10)

            # print(line_new)

            blank_im = np.zeros((height, width, 3), np.uint8)
            def draw_curve(list_lines, img, i):
                for i, e in enumerate(list_lines):
                    if e[10] == 12:
                        # Blue is a curvature
                        color = (255, 0, 0)
                    elif e[10] == 13:
                        # Green is a discontinuity
                        color = (0, 255, 0)
                    else:
                        # Red is a 'hole' line
                        color = (0, 0, 255)
                    x1 = int(e[1])
                    y1 = int(e[0])
                    x2 = int(e[3])
                    y2 = int(e[2])
                    cv2.line(img, (x1, y1), (x2, y2), color, 2)
                # cv2.imshow("curvatures", img)
                cv2.imshow("Curvature%d" % numImg, img)
                cv2.imwrite("Curvature%d%d.png" % (numImg, i), img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            draw_curve(line_new, blank_im, j)


            def draw_label(list_lines, img, i):
                for i, e in enumerate(list_lines):
                    # left of disc
                    if e[12] == 1:
                        # Teal
                        color = (255, 255, 0)
                    # right of disc
                    elif e[12] == 2:
                        # Orange
                        color = (0, 165, 255)
                    # convex/left of curv
                    elif e[12] == 3:
                        # Pink
                        color = (194, 89, 254)
                    # convex/right
                    elif e[12] == 32:
                        # lavender
                        color = (255, 182, 193)
                        # color = (194, 89, 254)
                    # concave of curv
                    elif e[12] == 4:
                        # Purple
                        color = (128, 0, 128)
                    # Remove
                    elif e[12] == -1:
                        # Yellow
                        color = (0, 255, 255)
                    else:
                        # Red is a 'hole' line
                        color = (0, 0, 255)
                    x1 = int(e[1])
                    y1 = int(e[0])
                    x2 = int(e[3])
                    y2 = int(e[2])
                    cv2.line(img, (x1, y1), (x2, y2), color, 2)
                # cv2.imshow("Labels", img)
                cv2.imshow("Label%d" % numImg, img)
                cv2.imwrite("Label%d%d.png" % (numImg, i), img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            # print(edgelist[j], "edgelist[j]")
            line_new = lc.label_curves(img, line_new, ListPoint_new, edgelist[j])

            blank_im = np.zeros((height, width, 3), np.uint8)
            draw_label(line_new, blank_im, j)
    

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
        #     print(line_new.shape, 'before')
            delet_these = np.where(np.logical_or(line_new[:, 12] == 4, line_new[:, 12] == -1))
            line_new = np.delete(line_new, delet_these, axis=0)
            # print(line_new.shape, 'after')

            list_pair = line_match(line_new, param, blank_image)

            print('List pair:', list_pair)
            blank_im = np.zeros((height, width, 3), np.uint8)
            util.draw_listpair(list_pair, line_new, final_im)