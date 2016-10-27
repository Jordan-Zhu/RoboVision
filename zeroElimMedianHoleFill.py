import numpy as np
from ZeroElimMedianFilter import zero_elim_median_filter


def zeroElimMedianHoleFill(im):
    r = zero_elim_median_filter(im)
    has_holes = ~np.all(np.all(r))
    while has_holes:
        r = zero_elim_median_filter(r)
        has_holes = ~np.all(np.all(r))
    return r
