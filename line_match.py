import numpy as np
from check_overlap import check_overlap
from relative_pos import relative_pos
from distance2d import distance2d

# Input: LineInteresting, P(Parameters)
# Output: ListPair
def line_match(LineInteresting, P):

    rowsize = LineInteresting.shape[0]
    m_mat = np.zeros((rowsize, rowsize))
    d_mat = np.zeros((rowsize, rowsize))
    ListPair = [0, 0]

    i = 0
    cnt = 1

    while i <= rowsize:
        if LineInteresting[i, 4] > P["Cons_Lmin"]:
            j = i + 1
            while j <= rowsize:
                if LineInteresting[j, 4] > P["Cons_Lmin"]:
                    if abs(LineInteresting[i, 6] - LineInteresting[j, 6]) <= P["Cons_Lmin"] or (abs(LineInteresting(i,7)-LineInteresting(j,7))>= (180-P["Cons_AlphaD"])):
                        m_mat[i, j] = 1
                        d = distance2d(LineInteresting[i, :], LineInteresting[j, :])
                        d_mat[i, j] = d
                        if d < P["Cons_Dmax"] and d > P["Cons_Dmin"]:
                            flag_overlap = check_overlap(LineInteresting[i, :], LineInteresting[j,:])
                            if flag_overlap:
                                flag_relative = relative_pos(LineInteresting[i, :],LineInteresting[j,:])
                                if flag_relative:
                                    ListPair[cnt, :] = [LineInteresting[i, 7], LineInteresting[j, 7]]
                                    cnt += 1
                                # end if
                            # end if
                        # end if
                    j += 1
                    # end if
            # end while
        # end if
    i += 1
    # end while

    return ListPair