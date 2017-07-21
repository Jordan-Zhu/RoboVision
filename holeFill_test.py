import cv2
import numpy as np
import util as util


if __name__ == '__main__':
    im = cv2.imread('img/learn1.png', -1)
    mask = np.ma.masked_where(im <= 0, im)
    cv2.imshow("mask", mask)
    cv2.waitKey()
    cv2.destroyAllWindows()