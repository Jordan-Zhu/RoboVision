import cv2
import numpy as np
import math

def zeroElimMedianFilter (im):
    rows = im.shape[0]
    cols = im.shape[1]
    r = np.zeros((rows, cols))
    im = np.lib.pad(im, ((2, 2), (2, 2)), 'edge')

    for m in range (0,rows-1):
        for n in range (0,cols-1):