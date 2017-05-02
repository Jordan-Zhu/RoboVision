import numpy as np


def availablepixels(rp, cp, edgeNo):

	# If edgeNo not supplied set to 0 to allow all adjacent functions
	# to be returned.
	if edgeNo == -1:
		edgeNo = 0

	ra = []
	ca = []
	rj = []
	cj = []

	# Row and column offsets for the eight neighbors of a point.
	roff = [-1 0 1 1 1 0 -1 -1]
	coff = [-1 -1 -1 0 1 1 1 0]

	row = rp + roff
	col = cp + coff

	# Find indices of arrays of r and c that are within the image bounds.
	ind = np.where((r >= 1 & r <= ROWS) & (c >= 1 & c <= COLS))

	# A pixel is available for linking if its value is 1 or it is a 
	# junction that has not been labeled -edgeNo
	for i in ind:
		if EDGEIM[r[i[0]], c[i[0]]] == 1 and not JUNCT[r[i[0]], c[i[0]]]:
			ra = [ra, r[i]]
			ca = [ca, c[i]]
		elif EDGEIM[r[i[0]], c[i[0]]] != -edgeNo and JUNCT[r[i[0]], c[i[0]]]:
			rj = [rj, r[i]]
			cj = [cj, c[i]]