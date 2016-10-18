# from loadmat import loadmat
import numpy as np
from line_match import line_match
import scipy.io as sio


if __name__ == '__main__':
    data1 = sio.loadmat('LineInteresting_in.mat')
    LineInteresting = np.array(data1['LineInteresting'])

    data2 = sio.loadmat('Parameter.mat')
    Parameter = data2['P']

    out = sio.loadmat('ListPair_out.mat')
    ListPair = out['ListPair']

    a = line_match(LineInteresting, Parameter)
    print(a)
    # print("Line interesting\n", LineInteresting)
    for item, value in enumerate(LineInteresting):
        print(item, value)

    # print(int(Parameter["Cons_Lmin"]))
    # print(int(Parameter["Cons_AlphaD"]))
    # print(int(Parameter["Cons_Dmax"]))
    # print(int(Parameter["Cons_Dmin"]))

    # print("Parameter\n", Parameter)

    # rowsize = LineInteresting.shape[0] - 1
    # for i in range(0, rowsize):
    #     j = i + 1
    #     for j in range(j, rowsize):
    #         if(ListPair[:][:] == LineInteresting[i, 7] and ListPair[:][:] == LineInteresting[j, 7]):
    #             print("Lines ", i, "and ", j, "were chosen")

    print("MATLAB List pair:\n", ListPair)