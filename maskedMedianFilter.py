import numpy as np
import math


def masked_median_filter(im):
    r = im
    im = np.lib.pad(im, ((2, 2), (2, 2)), 'edge')
    mask = np.nonzero(im)
    rows = mask.shape[0]
    cols = mask.shape[1]
    i = np.arange(-2, 3)
    j = np.arange(-2, 3)
    # for x in range(0, rows):
