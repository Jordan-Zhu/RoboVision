import cv2 as cv2
import numpy as np
import util as util
import curv_disc as cd
import depth_disc as dd
import skimage
import copy
import random

def create_img(mat):
    blank_image = np.zeros((mat.shape[0], mat.shape[1], 3), np.uint8)
    # print(blank_image.shape)
    mask = np.array(mat * 255, dtype=np.uint8)
    masked = np.ma.masked_where(mask <= 0, mask)

    return mask
    

def edge_detect(depth):
    curve_disc, curve_con = cd.curve_discont(depth)
    depth_disc, depth_con = dd.depth_discont(depth)

    # squeeze_ndarr(curve_con)
    # squeeze_ndarr(depth_con)

    # combine both images
    dst = (np.logical_or(curve_disc, depth_disc)).astype('uint8')
    #print(dst, "dst")
    #print(type(dst), "type")
    #print("dst", dst)

    #checking = checking.astype('uint8')
    dst = create_img(dst)
    util.showimg(dst, "Depth + Discontinuity")

    """

        img = dst
        edges = cv2.Canny(dst,50, 150, apertureSize = 3)

        lines = cv2.HoughLinesP(edges,1,np.pi/30,20, 10, 20)

        for x in range(len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                cv2.line(img,(x1,y1),(x2,y2),(0,127,127),2)

        cv2.imshow("someshit2", img)

    """





    """  
      _, markers = cv2.connectedComponents(dst)
      print(np.amax(markers), "marker numbers")
      checking = cv2.connectedComponents(dst)
  
      #print(checking)
      #print("checking")
  
      markers = skimage.color.label2rgb(markers)
      cv2.imshow('markers', markers)
      print(markers)"""



    skel_dst = util.morpho(dst)
    out = mask_contours(create_img(skel_dst))
    res = []
    # print(np.squeeze(out[0]))
    # print(out[0][0])
    for i in range(len(out)):
        # Add the first point to the end so the shape closes
        current = np.squeeze(out[i])
        # print('current', current)
        # print('first', out[i][0])
        if current.shape[0] > 2:
            # res.append(np.concatenate((current, out[i][0])))
            # print(res[-1])
            res.append(current)
        # print(np.concatenate((np.squeeze(out[i]), out[i][0])))

    res = np.array(res)
    util.sqz_contours(res)
    # squeeze_ndarr(res)

    #What is the point of this line
    #dst = util.find_contours(create_img(skel_dst), cv2.RETR_EXTERNAL)

    #util.showimg(dst, "Depth + Discontinuity2")


    return curve_disc, curve_con, depth_disc, depth_con, res


def mask_contours2(im):
    # showimg(im)
    height = im.shape[0]
    width = im.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8)
    im2, contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = []
    cntrs.append(contours)
    cntr1 = copy.deepcopy(contours)


    for eachC in range(len(contours)):
         cnt = contours[eachC]
         epsilon = 0.0005*cv2.arcLength(cnt,True)
         approx = cv2.approxPolyDP(cnt,epsilon,True)
         contours[eachC] = approx
                     

    cntr1 = contours
    cv2.drawContours(im, contours, -1, (0, 0, 0), 1, 8)
    cv2.drawContours(blank_image, contours, -1, (0, 255, 0), 1, 8)
    cv2.imshow("CONTOURS", blank_image)
    """"   print(len(contours), "contours")
                            print(len(contours[0]), "contours")"""

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    blank_image = np.zeros((height, width, 3), np.uint8)
    im2, contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(hierarchy)
    # print(contours)
    # draw_contours(blank_image, contours)
    cntr2 = copy.deepcopy(contours)


    for eachC in range(len(contours)):
        cnt = contours[eachC]
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        contours[eachC] = approx
            

    cv2.drawContours(blank_image, contours, -1, (0, 255, 0), 1, 8)
    cv2.imshow("CONTOURS 2", blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cntrs.append(contours)
    cntr2 = contours

    return cntr1 + cntr2

def mask_contours(im):
    # showimg(im)
    height = im.shape[0]
    width = im.shape[1]
    #print(im)
    blank_image = np.zeros((height, width, 3), np.uint8)
    #im = tryConnected(im)
    im2, contours, hierarchy = cv2.findContours(im, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print(hierarchy, "hierarchy")
    print(len(contours), "len contours")
    print(len(contours[0]), "0")
    print(len(contours[1]), "1")
    print(len(contours[2]), "2")
    cntrs = []
    cntrs.append(contours)
    #cv2.drawContours(im, contours, -1, (0, 0, 0), 1, 8)
    for x in range(len(contours)):
        for z in range(x+1, len(contours)):
            for y in range(len(contours[x])):
                for j in range(len(contours[z])):
                    if(contours[x][y][0][0] == contours[z][j][0][0] and
                        contours[x][y][0][1] == contours[z][j][0][1]):
                        print(contours[x][y][0], "same")

    for x in range(len(contours)):
        randC = random.uniform(0, 1)
        randB = random.uniform(0,1)
        randA = random.uniform(0,1)
        cv2.drawContours(blank_image, contours, x, (int(randA*255), int(randB*255), int(randC*255)), 1, 8)
    cv2.imshow("CONTOURS", blank_image)
    print(len(contours), "contours")
    print(len(contours[0]), "contours")
    cv2.imwrite("checking_new.png", blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cntr1 = contours
    print(len(cntr1), "len cntr1")
    print(len(cntr1[0]), "len cntr1")




    """
    blank_image = np.zeros((height, width, 3), np.uint8)
    im2, contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(hierarchy, "heirarchy")"""
    # print(hierarchy)
    # print(contours)
    # draw_contours(blank_image, contours)
            

    #cv2.drawContours(blank_image, contours, -1, (0, 255, 0), 1, 8)
    #cv2.imshow("CONTOURS 2", blank_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cntrs.append(contours)
    #cntr2 = contours
    #print(len(cntr2), "len cntr2")
    #print(len(cntr2[0]), "len cntr2")


    return cntr1


def tryConnected(img):
    cv2.imshow("someimage", img)
    _, markers = cv2.connectedComponents(img)
    markers = skimage.color.label2rgb(markers)
    print(markers)
    ret, thresh = cv2.threshold(markers, 127, 255, 0)
    img = cv2.bitwise_not(thresh)   
    """for x in range(len(markers)):
                    for y in range(len(markers)):
                        if(markers[x][y] > 1):
                            markers[x][y] = 1
                            print("yes working")"""
    #print(markers)
    #print(img)
    cv2.imshow('markers', markers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    return markers