import numpy as np
import scipy.ndimage as ndimage


def findendsjunctions(edge_im):
	# Set up look up table to find junctions. To do this
	# we use the functions defined to test that the center
	# pixel within a 3x3 neighborhood is a junction.
	junctions = ndimage.generic_filter(edge_im, junction, size=(3,3))

	ends = ndimage.generic_filter(edge_im, ending, size=(3,3))



def junction(x):

	a = [x[1], x[2], x[3], x[6], x[9], x[8], x[7], x[4]].T
	b = [x[2], x[3], x[6], x[9], x[8], x[7], x[4], x[1]].T
	crossings = sum(abs(a - b))

	b = x[5] and crossings >= 6
	return b


def ending(x):
	a = [x[1], x[2], x[3], x[6], x[9], x[8], x[7], x[4]].T
	b = [x[2], x[3], x[6], x[9], x[8], x[7], x[4], x[1]].T
    crossings = sum(abs(a - b));
    
    b = x[5] and crossings == 2
    return b