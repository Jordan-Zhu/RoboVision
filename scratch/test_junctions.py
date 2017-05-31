import cv2
import numpy as np
from python.findendsjunctions import findendsjunctions
from timeit import default_timer as timer


if __name__ == '__main__':
    im = cv2.imread('bw_lambda.png')
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    start = timer()
    rj, cj, re, ce = findendsjunctions(gray)
    end = timer()
    print('time elapsed:', end - start, 'seconds')

    print('rj', rj)
    print('cj', cj)
    print('re', re)
    print('ce', ce)

    junc = []
    for i in range(len(rj)):
        junc.append([cj[i], rj[i]])

    ends = []
    for i in range(len(re)):
        ends.append([ce[i], re[i]])

    height = im.shape[0]
    width = im.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8)

    for i, e in enumerate(junc):
        x1 = int(e[0])
        y1 = int(e[1])
        cv2.line(im, (x1, y1), (x1, y1), (0, 255, 0), 1)

    for i, e in enumerate(ends):
        x1 = int(e[0])
        y1 = int(e[1])
        cv2.line(im, (x1, y1), (x1, y1), (0, 0, 255), 1)

    cv2.namedWindow('Junctions and Ends', cv2.WINDOW_NORMAL)
    cv2.imshow('Junctions and Ends', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()