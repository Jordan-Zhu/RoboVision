import cv2
import numpy as np
import python.findendsjunctions as fj
import python.trackedge as te
from timeit import default_timer as timer


if __name__ == '__main__':
    im = cv2.imread('bw_lambda.png')
    edge_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    start = timer()
    rj, cj, re, ce = fj.findendsjunctions(edge_im)
    end = timer()
    print('time elapsed:', end - start, 's')

    print('rj', rj)
    print('cj', cj)
    print('re', re[0])
    print('ce', ce[0])

    junct = np.zeros_like(edge_im)
    print('JUNCT shape', junct.shape)
    for i in range(0, len(rj)):
        print('rj', rj[i])
        print('cj', cj[i])
        junct[rj[i], cj[i]] = 1

    edge_points, end_type = te.trackedge(edge_im, junct, re[0], ce[0], 1)
    print('edge_points shape', edge_points.shape)
    print('edge_points:\n', edge_points)