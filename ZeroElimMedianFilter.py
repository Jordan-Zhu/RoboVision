import numpy as np
import math


def zero_elim_median_filter (im):
    rows = im.shape[0]
    cols = im.shape[1]
    r = np.zeros((rows, cols))
    im = np.lib.pad(im, ((2, 2), (2, 2)), 'edge')

    i = np.arange(-2, 3)
    for m in range (0,rows):
        for n in range (0,cols):
            a = im[m+i+3, n+i+3]
            print(a.shape)
            a = a.flatten()
            a = np.where(a != 0)
            print(a)
            if a:
                a = np.sort(a)
                print(a)
                r[m+1, n+1] = a[math.ceil(a.shape[1] / 2)]
            else:
                r[m+1, n+1] = im[m+1, n+1]
            # end-else
        # end-for
    # end-for
    return r
# end
