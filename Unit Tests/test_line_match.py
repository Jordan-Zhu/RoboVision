from loadmat import loadmat
import numpy as np
from line_match import line_match
import scipy.io as sio


if __name__ == '__main__':
    data1 = sio.loadmat('LineInteresting_in.mat')
    LineInteresting = np.array(data1['LineInteresting'])

    data2 = sio.loadmat('Parameter.mat')
    Parameter = list(data2['P'])

    # I haven't been able to figure out how to get the list into a dictionary
    # or another way to use it
    print(Parameter)
    line_match(LineInteresting, Parameter)
