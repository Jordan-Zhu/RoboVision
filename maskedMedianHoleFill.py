import numpy as np
from ZeroElimMedianFilter import zero_elim_median_filter


def zeroElimMedianHoleFill(im):
    r = masked_median_filter(im, im == 0)
    has_holes = ~np.all(np.all(r))
    while has_holes:
        r = masked_median_filter(r, r == 0)
        has_holes = ~np.all(np.all(r))
    return r