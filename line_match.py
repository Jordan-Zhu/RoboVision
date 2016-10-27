from check_overlap import check_overlap
from relative_pos import relative_pos
from distance2d import distance2d

# Written 10/27/2016

# Input: LineInteresting, P(Parameters)
# Output: ListPair


def line_match(LineInteresting, P):

    rowsize = LineInteresting.shape[0]
    ListPair = []

    for i in range(0, rowsize):
        # If the length of the line is larger than min
        if LineInteresting[i, 4] > int(P["Cons_Lmin"]):
            j = i + 1
            for j in range(j, rowsize):
                # If length of line is larger than min
                if LineInteresting[j, 4] > int(P["Cons_Lmin"]):
                    # If it satisfies the slope constraint
                    if abs(LineInteresting[i, 6] - LineInteresting[j, 6]) <= int(P["Cons_AlphaD"]) or (abs(LineInteresting[i,6] - LineInteresting[j,6]) >= (180-int(P["Cons_AlphaD"]))):
                        d = distance2d(LineInteresting[i, :], LineInteresting[j, :])
                        # If it satisfies the maximum distance
                        if int(P["Cons_Dmin"]) < d < int(P["Cons_Dmax"]):
                            isOverlapping = check_overlap(LineInteresting[i, :], LineInteresting[j, :])
                            if isOverlapping:
                                flag_relative = relative_pos(LineInteresting[i, :],LineInteresting[j,:])

                                if flag_relative:
                                    # print("Lines ", i, "and ", j, "were chosen")
                                    ListPair.append([LineInteresting[i, 7], LineInteresting[j, 7]])
                                    # cnt += 1
                                # end if
                            # end if
                        # end if
                    # j += 1
                    # end if
            # end while
        # end if
        # i += 1
        # print("i =", i)
    # end while

    return ListPair