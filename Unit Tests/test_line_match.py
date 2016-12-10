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
    print("Python List Pair:\n", a)
    # print("Line interesting\n", LineInteresting)
    # for item, value in enumerate(LineInteresting):
    #     print(item, value)


    newLineInteresting = []
    for i in range(0, len(LineInteresting)):
        newLineInteresting.append([LineInteresting[i][0], LineInteresting[i][1], LineInteresting[i][2],
                                   LineInteresting[i][3], LineInteresting[i][4], LineInteresting[i][5],
                                   LineInteresting[i][6], LineInteresting[i][7], LineInteresting[i][8],
                                   LineInteresting[i][9], LineInteresting[i][10], LineInteresting[i][11]])

    # print("Line Interesting:")
    # for i, line in enumerate(newLineInteresting):
    #     print(i, ". ", line[7])

    # print(int(Parameter["Cons_Lmin"]))
    # print(int(Parameter["Cons_AlphaD"]))
    # print(int(Parameter["Cons_Dmax"]))
    # print(int(Parameter["Cons_Dmin"]))

    # print("Parameter\n", Parameter)

    newList = []
    for i in range(0, len(ListPair)):
        newList.append([ListPair[i][0], ListPair[i][1]])
    print("MATLAB List pair:")
    print(newList)

    # rowsize = LineInteresting.shape[0] - 1
    # for i in range(0, rowsize):
    #     j = i + 1
    #     for j in range(j, rowsize):
    #         if(ListPair[:][:] == LineInteresting[i, 7] and ListPair[:][:] == LineInteresting[j, 7]):
    #             print("Lines ", i, "and ", j, "were chosen")

    # print("MATLAB List pair:\n", ListPair)
