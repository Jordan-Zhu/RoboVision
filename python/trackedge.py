

def trackedge(rstart, cstart, edge_no, r2, c2, avoidJunctions):
	edge_points = [rstart, cstart]	# Start a new list for this edge.
	EDGEIM(rstart, cstart) = -edge_no  # Edge points in the iamge are 
									   # encoded by -ve of their edgeNo.

	# If the second point has been supplied, add it to the track
	# and set the path direction.
	if r2 && c2 != -1:
		edge_points = [edge_points, r2, c2]
		EDGEIM[r2, c2] = -edge_no

		# Initialize direction vector of path and set the current point on 
		# the path
		dirn = unitvector([r2 - rstart, c2 - cstart])
		row = r2
		col = c2
		preferred_dir = 1
	else:
		dirn = [0, 0]
		row = rstart
		col = cstart

	
	# Find all pixels we could link to
	[r_avail, c_avail, r_junc, c_junc] = availablepixels(row, col, edge_no)

	while r_avail or r_junc:
		# First see if we can link to a junction. Choose the junction
		# that results in a move that is as close as possible to dir. 
		# If we have no preferred direction, and there is a choise,
		# link to the closest junction.
		# 
		if r_junc and 