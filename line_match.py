import numpy as np
from check_overlap import check_overlap
from relative_pos import relative_pos
from distance2d import distance2d

# Input: LineInteresting, P(Parameters)
# Output: ListPair
def line_match(LineInteresting, P):

    rowsize = LineInteresting.shape[0] - 1
    m_mat = np.zeros((rowsize, rowsize))
    d_mat = np.zeros((rowsize, rowsize))
    ListPair = []

    # i = 0
    # cnt = 0

    for i in range(0, rowsize):
        # If the length of the line is larger than min
        if LineInteresting[i, 4] > int(P["Cons_Lmin"]):
            j = i + 1
            for j in range(j, rowsize):
                # If length of line is larger than min
                if LineInteresting[j, 4] > int(P["Cons_Lmin"]):
                    # If it satisfies the slope constraint
                    if abs(LineInteresting[i, 6] - LineInteresting[j, 6]) <= int(P["Cons_AlphaD"]) or (abs(LineInteresting[i,6] - LineInteresting[j,6]) >= (180-int(P["Cons_AlphaD"]))):
                        # print("i =", i, "j =", j)
                        m_mat[i, j] = 1
                        d = distance2d(LineInteresting[i, :], LineInteresting[j, :])
                        d_mat[i, j] = d
                        # If it satisfies the maximum distance
                        if d < int(P["Cons_Dmax"]) and d > int(P["Cons_Dmin"]):
                            flag_overlap = check_overlap(LineInteresting[i, :], LineInteresting[j,:])
                            # print("Flag overlap =", flag_overlap)
                            if flag_overlap:
                                flag_relative = relative_pos(LineInteresting[i, :],LineInteresting[j,:])

                                if flag_relative:
                                    # print(LineInteresting[i, 7], " + ", LineInteresting[j, 7])
                                    # Append this line to ListPair
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