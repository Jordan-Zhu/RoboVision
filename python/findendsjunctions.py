import numpy as np
import scipy.ndimage as ndimage


def findendsjunctions(edge_im):
	# Set up look up table to find junctions. To do this
	# we use the functions defined to test that the center
	# pixel within a 3x3 neighborhood is a junction.
	junctions = ndimage.generic_filter(edge_im, junction, size=(3,3))
	[rj, cj] = np.nonzero(junctions)

	ends = ndimage.generic_filter(edge_im, ending, size=(3,3))
	[re, ce] = np.nonzero(ends)

	return rj, cj, re, ce



def junction(x):

	a = np.array([x[1], x[2], x[3], x[6], x[9], x[8], x[7], x[4]])
	b = np.array([x[2], x[3], x[6], x[9], x[8], x[7], x[4], x[1]])
	crossings = np.sum(np.absolute(np.subtract(a, b)))

	return x[5] and crossings >= 6


def ending(x):
	
	a = np.array([x[1], x[2], x[3], x[6], x[9], x[8], x[7], x[4]])
	b = np.array([x[2], x[3], x[6], x[9], x[8], x[7], x[4], x[1]])
	crossings = np.sum(np.absolute(np.subtract(a, b)))
    
	return x[5] and crossings == 2