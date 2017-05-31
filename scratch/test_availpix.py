import cv2
import numpy as np
import python.findendsjunctions as fj
import python.availablepixels as ap
from timeit import default_timer as timer


if __name__ == '__main__':
    # global EDGEIM
    # global JUNCT
    # global ROWS
    # global COLS

    im = cv2.imread('bw_lambda.png')
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    edge_im = gray
    rows, cols = edge_im.shape

    start = timer()
    rj, cj, re, ce = fj.findendsjunctions(gray)
    end = timer()
    print('time elapsed:', end - start, 's')

    print('rj', rj)
    print('cj', cj)
    print('re', re)
    print('ce', ce)

    # Create a sparse matrix to mark junction locations.
    # This makes junction testing much faster. A value of 1 indicates a junction,
    # and a value of 2 indicates we have visited the junction.
    junct = np.zeros_like(edge_im)
    print('JUNCT shape', junct.shape)
    for i in range(0, len(rj)):
        print('rj', rj[i])
        print('cj', cj[i])
        junct[rj[i], cj[i]] = 1

    rstart = re[0]
    cstart = ce[0]

    start = timer()
    ra, ca, rj, cj = ap.availablepixels(edge_im, rows, cols, junct, rstart, cstart)
    end = timer()
    print('time elapsed:', end - start, 's')

    print('ra', ra)
    print('ca', ca)
    print('rj', rj)
    print('cj', cj)
