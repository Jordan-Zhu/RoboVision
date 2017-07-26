from check_overlap import check_overlap
from relative_pos import relative_pos
from distance2d import distance2d
import copy
import random as rand
import cv2
import numpy as np
# Written 10/27/2016

# Input: LineInteresting, P(Parameters)
# Output: ListPair


def line_match(LineInteresting, P, blankimage):
    # Constants
    minlen = 20
    delta_angle = int(P["Cons_AlphaD"])
    min_dist = 10
    max_dist = 200
    print(minlen, delta_angle, min_dist, max_dist, "parameter values")

    rowsize = LineInteresting.shape[0]
    #Number of lines in this contour

    ListPair = []
    for i in range(0, rowsize): 
        ##Checks and makes sure they're of a certain length and that they're not concave
        if(LineInteresting[i, 4] > minlen) and (LineInteresting[i, 12] != 4):
            print(i, "less than min and not concave")
            for j in range(i+1, rowsize):
                ##Checks and makes sure the one compared to is of a certain length and that they're not concave
                if(LineInteresting[j, 4] > minlen) and (LineInteresting[j, 12] != 4):
                    # If it is in range of the slope constraint
                    if(abs(LineInteresting[i, 4]-LineInteresting[j, 4]) < .5*(max(LineInteresting[j, 4], LineInteresting[i, 4]))):
                        if abs(LineInteresting[i, 6] - LineInteresting[j, 6]) <= delta_angle or (abs(LineInteresting[i,6] - LineInteresting[j,6]) >= (180-delta_angle)):

                            blank_image = np.zeros((blankimage.shape[0], blankimage.shape[1], 3), np.uint8)

                            color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
                            cv2.line(blank_image, (int(LineInteresting[i][1]), int(LineInteresting[i][0])), (int(LineInteresting[i][3]), int(LineInteresting[i][2])), color, thickness=1)
                            cv2.line(blank_image, (int(LineInteresting[j][1]), int(LineInteresting[j][0])), (int(LineInteresting[j][3]), int(LineInteresting[j][2])), (255,255,255), thickness=1)
                            cv2.imshow("lines%d"%i, blank_image)
                            cv2.waitKey(0)
                            print(LineInteresting[i, 6], LineInteresting[j, 6], "these are the anglessssss")

                            if(LineInteresting[i, 12] != LineInteresting[j, 12]):
                                d = distance2d(LineInteresting[i, :], LineInteresting[j, :])
                                print(d, "d valueeeeeeeeeeeeeeeeeeeee")
                                #ddd = distance3d(LineInteresting[i], LineInteresting[j])
                                # The line cannot be too short or too long
                                if min_dist < d < max_dist:
                                    ###Checking overlap, no longer necessary due to only looking within a contour##
                                    """
                                    isOverlapping = check_overlap(LineInteresting[i, :], LineInteresting[j, :])
                                    if isOverlapping:
                                        flag_relative = relative_pos(LineInteresting[i, :],LineInteresting[j,:])
                                    
                                    if flag_relative:
                                        ListPair.append([LineInteresting[i, 7].astype(int), LineInteresting[j, 7].astype(int)])"""
                                    ListPair.append([i,j])
                                        
                                        # end if
                                    # end if
                            # end if
                        # end if
                # end while
            # end if
        # end while
    print(ListPair, "listpair")

    return ListPair