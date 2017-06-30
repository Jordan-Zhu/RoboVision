import numpy as np
import scipy.io as sio


if __name__ == '__main__':
    data = sio.loadmat('in_out_ML_2.mat')
    ListPointC = data['ListPointC']

    pt1 = 62
    pt2 = 64
    lind1 = 14498
    lind2 = 14998

    lp1 = ListPointC[pt1]
    lp2 = ListPointC[pt2]
    startpt1 = np.where(lp1[0] == lind1)[0]
    startpt2 = np.where(lp1[0] == lind2)[0]
    startpt3 = np.where(lp2[0] == lind1)[0]
    startpt4 = np.where(lp2[0] == lind2)[0]

    if not startpt1:
        line_start = lp2[0] # list([lp2])
        line_end = lp1[0] # list([lp1])

        if startpt3 > 0:
            line_start = line_start[::-1] # list(reversed(line_start))
        if startpt2 == 0:
            line_end = line_end[::-1] # list(reversed(line_end))
    else:
        line_start = lp1[0] # list([lp1])
        line_end = lp2[0] # list([lp2])

        if startpt1 > 0:       # startpt1[0] > 0:
            line_start = line_start[::-1] # list(reversed(line_start))
        if startpt4 == 0:      # startpt4[0] == 0:
            line_end = line_end[::-1] # list(reversed(line_end))

    print('line start:', line_start[0:-1], '\nline end:', line_end)
    merged = np.r_[line_start[0:-1], line_end]
    print('concatenate:', merged.shape)
    # print(ListPointC[553])
    temp = list(ListPointC)
    print(temp[0])
    temp.append(merged)
    # listpt = np.append(ListPointC, [merged])
    print(temp)

