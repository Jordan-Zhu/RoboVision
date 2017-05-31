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
    print('re', re)
    print('ce', ce)

    junct = np.zeros_like(edge_im)
    print('JUNCT shape', junct.shape)
    for i in range(0, len(rj)):
        print('rj', rj[i])
        print('cj', cj[i])
        junct[rj[i], cj[i]] = 1

    edge_no = 0
    edge_list = []
    for i in range(len(re)):
        if edge_im[re[i], ce[i]] == 255:    # edge point is unlabeled
            edge_no += 1
            edge_list[edge_no], end_type = te.trackedge(edge_im, junct, re[i], ce[i], edge_no)

    # edge_points, end_type = te.trackedge(edge_im, junct, re[0], ce[0], 1)
    print('edge_points:\n', edge_list)

    # for i, e in enumerate(edge_points):
    #     x1 = int(e[1])
    #     y1 = int(e[0])
    #     cv2.line(im, (x1, y1), (x1, y1), (0, 255, 0), 1)
    #
    # cv2.namedWindow('Track edge', cv2.WINDOW_NORMAL)
    # cv2.imshow('Track edge', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
