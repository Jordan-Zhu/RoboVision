#from check_overlap import check_overlap
from relative_pos import relative_pos
from distance2d import distance2d
import copy
import random as rand
import cv2
import numpy as np
# Written 10/27/2016

# Input: LineInteresting, P(Parameters)
# Output: ListPair

def line_match(LineInteresting, P):
    # Constants
    delta_angle = copy.deepcopy(P["delta_angle"])
    min_dist = copy.deepcopy(P["min_dist"])
    max_dist = copy.deepcopy(P["max_dist"])
    blank_image = copy.deepcopy(P["blank_image"])
    rowsize = LineInteresting.shape[0]
    #Number of lines in this contour

    list_pair = []
    matched_lines = []
    #Goes through every line
    for i in range(0, rowsize):

        #Compares it with every other line after it
        for j in range(i+1, rowsize):

            #Checking to make sure the smaller line is at least half the size of the larger line
            if(abs(LineInteresting[i, 4]-LineInteresting[j, 4]) < .5*(max(LineInteresting[j, 4], LineInteresting[i, 4]))):
                
                #Checking to make sure the angles are similar, within the threshold of each other
                if abs(LineInteresting[i, 6] - LineInteresting[j, 6]) <= delta_angle or (abs(LineInteresting[i,6] - LineInteresting[j,6]) >= (180-delta_angle)):

                    #checking to make sure that they are not the same kind of lines
                    #EX: not both discontinuity rights, or discontinuity lefts, or convex
                    if(LineInteresting[i, 12] != LineInteresting[j, 12]):
                        
                        #Checking to make sure they are not too far apart or two close together
                        d = distance2d(LineInteresting[i, :], LineInteresting[j, :])
                        if min_dist < d < max_dist:
                            list_pair.append([i,j])
                            matched_lines.append([LineInteresting[i], LineInteresting[j]])

    print(list_pair, "listpair")
    return list_pair, matched_lines