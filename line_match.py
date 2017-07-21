from check_overlap import check_overlap
from relative_pos import relative_pos
from distance2d import distance2d

# Written 10/27/2016

# Input: LineInteresting, P(Parameters)
# Output: ListPair


def line_match(LineInteresting, P):
    # Constants
    minlen = int(P["Cons_Lmin"])
    delta_angle = int(P["Cons_AlphaD"])
    min_dist = int(P["Cons_Dmin"])
    max_dist = int(P["Cons_Dmax"])
    print(minlen, delta_angle, min_dist, max_dist, "parameter values")

    rowsize = LineInteresting.shape[0]
    #Number of lines in this contour

    ListPair = []
    for i in range(0, rowsize): 
        ##Checks and makes sure they're of a certain length and that they're not concave
        if(LineInteresting[i, 4] > minlen) and (LineInteresting[i, 12] == 3):
            print(i, "less than min and not concave")
            for j in range(i+1, rowsize):
                ##Checks and makes sure the one compared to is of a certain length and that they're not concave
                if(LineInteresting[j, 4] > minlen) and (LineInteresting[j, 12] == 3):
                    # If it is in range of the slope constraint
                    print(j, "also less than min and not concave")
                    if abs(LineInteresting[i, 6] - LineInteresting[j, 6]) <= delta_angle or (abs(LineInteresting[i,6] - LineInteresting[j,6]) >= (180-delta_angle)):
                        print("past the angle")
                        if(LineInteresting[i, 12] != LineInteresting[j, 12]):
                            d = distance2d(LineInteresting[i, :], LineInteresting[j, :])
                            print(d, "d value")
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
                                ListPair.append([LineInteresting[i, 7].astype(int), LineInteresting[j, 7].astype(int)])
                                    
                                    # end if
                                # end if
                        # end if
                    # end if
            # end while
        # end if
    # end while
    print(ListPair, "listpair")

    return ListPair