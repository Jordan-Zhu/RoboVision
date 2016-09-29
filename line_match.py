# Input: LineInteresting, P
# Output: ListPair

import numpy as np

# TO-DO: write P as a python dictionary
def line_match(LineInteresting, P):
	rowsize = LineInteresting.shape[0]
	m_mat = np.zeros(rowsize, rowsize)
	d_mat = np.zeros(rowsize, rowsize)

	ListPair = [0, 0]
	i = 1
	count = 1
	while i <= rowsize:
		if LineInteresting[i, 4] > P.