import math

import numpy as np

from legacyCode import availablepixels as ap


def trackedge(edge_im, junct, rstart, cstart, edge_no, r2 = -1, c2 = -1, avoidJunctions = 0):
    rows, cols = edge_im.shape

    edge_points = [[rstart, cstart]]  # Start a new list for this edge.
    edge_im[rstart, cstart] = -edge_no  # Edge points in the image are
                                        # encoded by -ve of their edgeNo.

    preferred_dir = 0

    # If the second point has been supplied, add it to the track
    # and set the path direction.
    if r2 and c2 != -1:
        edge_points.append([r2, c2])
        edge_im[r2, c2] = -edge_no

        # Initialize direction vector of path and set the current point on
        # the path
        dirn = unitvector(np.array([r2 - rstart, c2 - cstart]))
        row = r2
        col = c2
        preferred_dir = 1
    else:
        dirn = [0, 0]
        row = rstart
        col = cstart

    # Find all pixels we could link to
    [r_avail, c_avail, r_junc, c_junc] = ap.availablepixels(edge_im, rows, cols, junct, row, col, edge_no)

    global rbest
    global cbest
    global dirnbest

    print('r_avail', r_avail)
    print('r_junc', r_junc)

    i = 0
    j = 0
    while i < len(r_avail) or j < len(r_junc):
        # First see if we can link to a junction. Choose the junction
        # that results in a move that is as close as possible to dir.
        # If we have no preferred direction, and there is a choice,
        # link to the closest junction.
        if (r_junc and avoidJunctions != 0) or (r_junc and not r_avail):
            # If we have a preferred direction, choose the junction
            # that results in a move that is as close as possible
            # to dirn.
            if preferred_dir != 0:
                dotp = [-math.inf, -math.inf]
                for n in range(0, len(r_junc)):
                    dirna = unitvector(np.array([r_junc[n] - row, c_junc[n] - col]))
                    dp = dirn * dirna
                    if np.greater(dp, dotp):
                        dotp = dp
                        rbest = r_junc[n]
                        cbest = c_junc[n]
                        dirnbest = dirna
            else:
                distbest = [math.inf, math.inf]
                for n in range(0, len(r_junc)):
                    dist = (r_junc[n] - row) + (c_junc[n] - col)
                    if np.less(dist, distbest):
                        rbest = r_junc[n]
                        cbest = c_junc[n]
                        distbest = dist
                        dirnbest = unitvector(np.array([r_junc[n] - row, c_junc[n] - col]))
                preferred_dir = 1
        else:
            dotp = [-math.inf, -math.inf]
            for n in range(0, len(r_avail)):
                dirna = unitvector(np.array([r_avail[n] - row, c_avail[n] - col]))
                dp = dirn * dirna
                # print('dirna', dirna)
                # print('dp', dp)
                # print('dotp', dotp)
                if all(np.greater(dp, dotp)):
                    dotp = dp
                    rbest = r_avail[n]
                    cbest = c_avail[n]
                    dirnbest = dirna
            avoidJunctions = 0

        # Append the best pixel to the edgelist and update the direction and edge image.
        row = rbest
        col = cbest
        edge_points.append([row, col])
        dirn = dirnbest
        edge_im[row, col] = -edge_no

        # If this point is a junction, exit here.
        if junct[row, col]:
            end_type = 1
            return edge_points, end_type
        else:
            # Get the next set of available pixels to link.
            [r_avail, c_avail, r_junc, c_junc] = ap.availablepixels(edge_im, rows, cols, junct, row, col, edge_no)
        i += 1
        j += 1

    end_type = 0    # Mark endpoints as being free, unless it is reset below.
    if len(edge_points) >= 4 and \
            abs(edge_points[0][0] - edge_points[-1][0]) <= 1 and \
            abs(edge_points[0][1] - edge_points[-1][1]) <= 1:
        edge_points.append(edge_points[0])
        end_type = 5    # Mark end type as being a loop.

    return edge_points, end_type


# Normalizes a vector to unit magnitude.
def unitvector(v):
    return np.divide(v, np.sqrt(v.dot(v)))
