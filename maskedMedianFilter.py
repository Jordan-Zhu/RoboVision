import numpy as np
import math


def masked_median_filter(im, mask):
    r = im
    im = np.lib.pad(im, ((2, 2), (2, 2)), 'edge')
    mask = np.array(np.nonzero(im))
    rows = mask.shape[0]
    cols = mask.shape[1]
    i = np.arange(-2, 3)
    j = np.arange(-2, 3)
    for x in range(0, rows):
        m = 1 # x.shape[0]
        n = 1 # x.shape[1]

        A = np.array(im[i+3, j+3])
        A = A.flatten()
        A = np.nonzero(A)
        if A:
            A = np.sort(A)
            temp = A.shape[1] / 2
            print(temp)
            r[m, n] = A[math.ceil(A.shape[1] / 2)]
