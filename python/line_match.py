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

    rowsize = LineInteresting.shape[0]
    ListPair = []

    for i in range(0, rowsize):
        if LineInteresting[i, 4] > minlen:
            j = i + 1
            for j in range(j, rowsize):
                if LineInteresting[j, 4] > minlen:
                    # If it is in range of the slope constraint
                    if abs(LineInteresting[i, 6] - LineInteresting[j, 6]) <= delta_angle or (abs(LineInteresting[i,6] - LineInteresting[j,6]) >= (180-delta_angle)):
                        d = distance2d(LineInteresting[i, :], LineInteresting[j, :])
                        # The line cannot be too short or too long
                        if min_dist < d < max_dist:
                            isOverlapping = check_overlap(LineInteresting[i, :], LineInteresting[j, :])
                            if isOverlapping:
                                flag_relative = relative_pos(LineInteresting[i, :],LineInteresting[j,:])

                                if flag_relative:
                                    ListPair.append([LineInteresting[i, 7].astype(int), LineInteresting[j, 7].astype(int)])
                                # end if
                            # end if
                        # end if
                    # end if
            # end while
        # end if
    # end while

    return ListPair