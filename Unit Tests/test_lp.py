import cv2
import numpy as np

from utility import normalize_depth

if __name__ == '__main__':
    img = cv2.imread('learn0.png', -1)
    img_size = img.shape
    points = [[164, 65], [164, 544], [165, 385], [568, 457]]
    print(img_size)

    listpoints = []
    for i, e in enumerate(points):
        # print(e[0])
        listpoints.append([np.ravel_multi_index((e[0], e[1]), img_size, order='C')])

    print(listpoints)
    res = []
    for i, lp in enumerate(listpoints):
        x1, y1 = np.unravel_index(lp, img_size, order='C')
        res.append([x1[0], y1[0]])

    print(res)