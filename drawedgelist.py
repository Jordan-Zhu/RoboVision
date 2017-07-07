# -------------------------------------------------------------------------------
# Name:        drawedgelist
# Purpose:     Plots pixels in edge lists.
# -------------------------------------------------------------------------------
#
# Usage:    drawedgelist(edgelist, rowscols, thickness)
#
# Arguments:
#   edgelist    - Array of arrays containing edgelists
#   rowscols    - Optional 2 element vector [rows cols] specifying the size
#                 of the image from which the edges were detected. Otherwise
#                 this defaults to the bounds of the line segment points.
#
# Author: Jordan Zhu
#


import sys
import cv2
import numpy as np
import random as rand


def drawedgelist(edgelist, *args, **kwargs):
    rowscols = kwargs.get('rowscols', None)

    if rowscols is None or rowscols == []:
        xmax = []
        ymax = []
        for idx, edges in enumerate(edgelist):
            xmax.append(np.amax(edges[:, 0]))
            ymax.append(np.amax(edges[:, 1]))

        # Get the rows and cols as the bounds of the line segment
        # with an added buffer on the sides.
        col = np.amax(xmax) + 10
        row = np.amax(ymax) + 10

        blank_image = np.zeros((row, col, 3), np.uint8)
    else:
        # Get the image size from the argument passed.
        blank_image = np.zeros((rowscols[0], rowscols[1], 3), np.uint8)

    for n, item in enumerate(edgelist):
        # Loop through the edge segment arrays in edgelist.
        idx = n # edgelist.index(n)
        x = edgelist[idx][:, 0]
        y = edgelist[idx][:, 1]

        blank_image = np.zeros((480, 640, 3), np.uint8)

        # Find the length of the arrays for index positions.
        # Either array can do since the points are in pairs.
        listlen = x.size - 1
        for i in range(listlen):
            # Draw the line segments.
            color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
            cv2.line(blank_image, (x[i], y[i]), (x[i + 1], y[i + 1]), color, thickness=1)
            # print('x', x[i], 'y', y[i], 'x1', x[i + 1], 'y1', y[i + 1])
            # cv2.imshow("Edge List", blank_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # Join the first and last line of the contour.
        # cv2.line(blank_image, (x[0], y[0]), (x[listlen], y[listlen]), (0, 255, 0), thickness=1)

    # Display the edge list.
    #cv2.namedWindow("Edge list", cv2.WINDOW_NORMAL)
    cv2.imshow("Edge list", blank_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print(edgelist)

"""
def drawedgelist2(edgelist, *args, **kwargs):
    listlen = x.size - 1
        for i in range(listlen):
            # Draw the line segments.
            color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
            cv2.line(blank_image, (x[i], y[i]), (x[i + 1], y[i + 1]), color, thickness=1)"""