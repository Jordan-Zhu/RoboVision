import matplotlib.pyplot as plt
import numpy as np
import cv2

def DrawLineFeature(linefeature, siz, im_name):
    xx = len(linefeature)
    blank_image = np.zeros(siz)

    for i in range(xx):
        x1 = int(linefeature[i][1])
        y1 = int(linefeature[i][0])
        x2 = int(linefeature[i][3])
        y2 = int(linefeature[i][2])
        cv2.line(blank_image, (x1,y1), (x2,y2), (2, 0, 200))
        # cv2.line(Id3,(x1,y1),(x2,y2),(0, 255, 255), thickness=2)

# def drawcontour(contour2,im_name):

    # for i in linefeature:
    #     for j in range(len(i)-1):
    #         cv2.line(blank_image,tuple(i[j]),tuple(i[j+1]),(2,0,200))

    cv2.imshow(im_name, blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def drawconvex(linefeature, siz, im_name):
    xx = len(linefeature)
    blank_image = np.zeros(siz)

    for i in range(xx):
        if linefeature[i][10] == 13:
            x1 = int(linefeature[i][1])
            y1 = int(linefeature[i][0])
            x2 = int(linefeature[i][3])
            y2 = int(linefeature[i][2])
            cv2.line(blank_image, (x1,y1), (x2,y2), (2, 0, 200))
        # cv2.line(Id3,(x1,y1),(x2,y2),(0, 255, 255), thickness=2)

# def drawcontour(contour2,im_name):

    # for i in linefeature:
    #     for j in range(len(i)-1):
    #         cv2.line(blank_image,tuple(i[j]),tuple(i[j+1]),(2,0,200))

    cv2.imshow(im_name, blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()