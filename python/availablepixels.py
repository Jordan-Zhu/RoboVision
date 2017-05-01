

def availablepixels(rp, cp, edgeNo):

	# If edgeNo not supplied set to 0 to allow all adjacent functions
	# to be returned.

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
	ind = -1

	# A pixel is available for linking if its value is 1 or it is a 
	# junction that has not been labeled -edgeNo
	for i in range(1,10):
		pass