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


    ######CHECK WHAT THE POINT OF THIS IS################
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

    out = np.asarray(out)
    print(len(out), print(out.shape), "out")
    print(len(res), print(res.shape), "res")
    # squeeze_ndarr(res)

    #What is the point of this line
    #dst = util.find_contours(create_img(skel_dst), cv2.RETR_EXTERNAL)

    #util.showimg(dst, "Depth + Discontinuity2")


    return curve_disc, curve_con, depth_disc, depth_con, res


def mask_contours(im):
    # showimg(im)
    height = im.shape[0]
    width = im.shape[1]
    #print(im)
    blank_image = np.zeros((height, width, 3), np.uint8)
    #im = tryConnected(im)
    im2, contours, hierarchy = cv2.findContours(im, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    print(type(contours), "type contours")
    print(hierarchy, "hierarchy")

    #cv2.drawContours(im, contours, -1, (0, 0, 0), 1, 8)
    #contours = fixOverlap(contours)

    """
    for x in range(len(contours)):
        epsilon = 0.005*cv2.arcLength(contours[x],True)
        approx = cv2.approxPolyDP(contours[x],epsilon,True)
        contours[x] = approx

    
    """
    contourCopy = copy.deepcopy(contours)
    totalDel = 0
    for x in range(len(contours)):
        area = cv2.contourArea(contours[x])
        print(area)
        #filters out a few contours that are too small to be of use
        # and also negative contours that wrap around things
        if(area < 500):
            del contourCopy[x-totalDel]
            totalDel += 1
        else:    
            randC = random.uniform(0,1)
            randB = random.uniform(0,1)
            randA = random.uniform(0,1)

            cv2.drawContours(blank_image, contours, x, (int(randA*255), int(randB*255), int(randC*255)), 1, 8)
    cv2.imshow("CONTOURS", blank_image)
    #cv2.imwrite("checking_2.png", blank_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    print(len(contourCopy), "len cntr1")
    print(len(contours), "len cntr1 old")




    """
    blank_image = np.zeros((height, width, 3), np.uint8)
    im2, contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(hierarchy, "heirarchy")
    # print(hierarchy)
    # print(contours)
    # draw_contours(blank_image, contours)
    
    for x in range(len(contours)):
        randC = random.uniform(0, 1)
        randB = random.uniform(0,1)
        randA = random.uniform(0,1)
        cv2.drawContours(blank_image, contours, x, (int(randA*255), int(randB*255), int(randC*255)), 1, 8)

    uniquePoints = np.unique(contours[2])
    print(len(uniquePoints), len(contours[2]), "unique")
    cv2.imshow("CONTOURS 2", blank_image)
    cv2.imwrite('checking_3.png', blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cntrs.append(contours)
    #cntr2 = contours
    #print(len(cntr2), "len cntr2")
    #print(len(cntr2[0]), "len cntr2")
    """

    return contourCopy


def fixOverlap(contours):
    #checking
    contours = np.asarray(contours)
    newContours = copy.deepcopy(contours)

    #Fixing the shape of contours, extra brackets for some reason
    for x in range(len(contours)):
        contoursShape = contours[x].shape
        contours[x] = np.reshape(contours[x], (contoursShape[0], 2), 0)
        print(contours[x].shape, "Contours[x].shape")
    
    #going through each contour except last one
    for x in range(len(contours)-1):
        #creates a mask to hold boolean values whether or not that point is unique to the array
        mask = np.ones(len(contours[x]), dtype=bool)

        #comparing it with each other contours
        for z in range(x+1, len(contours)):

            #checking each point in the first contour to see if it exists in the other contours
            for y in range(len(contours[x])):
                
                #checks if each row in z is equal to x,y. If any are equal, then it will delete the point from the x array
                #reference: https://stackoverflow.com/questions/33217660/checking-if-a-numpy-array-contains-another-array
                if((contours[z] == contours[x][y]).all(1).any()):
                        mask[[y]] = False                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

        #the mask contains an array such as [true, false, false]
        #with false on the points where there was an array with the same point
        newContours[x] = newContours[x][mask]
        print(len(newContours[x]), len(contours[x]),"deleteTotal")
        #print(deleteTotal, len(contours[x]), "comparison len")

            #print(total, len(contours[x]))
            #print(x, totalDelete,"totalDelete")

    return newContours

