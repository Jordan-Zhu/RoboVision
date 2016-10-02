import numpy as np

def line_match(LineInteresting, P):
	rowsize = LineInteresting.shape[0]
	m_mat = np.zeros(rowsize, rowsize)
	d_mat = np.zeros(rowsize, rowsize)

	i = 0
	cnt = 1
	while i <= N:
		if LineInteresting[i, 4] > P{Cons_Lmin}:
			j = i + 1
			while j <= rowsize:
				if LineInteresting[j, 4] > P{Cons_Lmin}:
					if abs(LineInteresting[i, 6] - LineInteresting[j, 6]) <= P{Cons_Lmin}||(abs(LineInteresting(i,7)-LineInteresting(j,7))>= (180-P{Cons_AlphaD})):
						m_mat[i, j] = 1
						d = distance2d(LineInteresting[i, :], LineInteresting[j, :])
						d_mat[i, j] = d
						if d < P{Cons_Dmax} && d > p{Cons_Dmin}:
							flag_overlap = check_overlap(LineInteresting[i, :], LineInteresting[j,:])
							if flag_overlap:
								flag_relative = relative_pos(LineInteresting[i, :],LineInteresting[j,:])
								if flag_relative:
									ListPair[cnt, :] = [LineInteresting[i, 7], LineInteresting[j, 7]]
									cnt += 1
					j += 1
		i += 1
	# end while

	return ListPair