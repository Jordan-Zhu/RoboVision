import numpy as np

# If edgeNo not supplied set to 0 to allow all adjacent functions
# to be returned.
def availablepixels(rp, cp, edge_im, rows, cols, junct, edge_no = 0):
    ra = []
    ca = []
    rj = []
    cj = []

    # Row and column offsets for the eight neighbors of a point.
    roff = np.array([-1, 0, 1, 1, 1, 0, -1, -1])
    coff = np.array([-1, -1, -1, 0, 1, 1, 1, 0])

    r = np.add(rp, roff)
    c = np.add(cp, coff)

    # Find indices of arrays of r and c that are within the image bounds.
    ind = np.where((0 <= r <= rows) & (0 <= c <= cols))

    # A pixel is available for linking if its value is 1 or it is a
    # junction that has not been labeled -edgeNo
    for i in ind:
        if edge_im[r[i[0]], c[i[0]]] == 255 and not junct[r[i[0]], c[i[0]]]:
            ra.append([ra, r[i]])
            ca.append([ca, c[i]])
        elif edge_im[r[i[0]], c[i[0]]] != -edge_no and junct[r[i[0]], c[i[0]]]:
            rj.append([rj, r[i]])
            cj.append([cj, c[i]])

    return ra, ca, rj, cj
