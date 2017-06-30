from check_overlap import check_overlap
from relative_pos import relative_pos
from distance2d import distance2d

# Written 10/27/2016
# Refactored 4/3/2017

# Input: LineInteresting, P(Parameters)
# Output: ListPair


def line_match(line_interesting, params):
    # Constants
    minlen = int(params["Cons_Lmin"])
    delta_angle = int(params["Cons_AlphaD"])
    min_dist = int(params["Cons_Dmin"])
    max_dist = int(params["Cons_Dmax"])

    list_pair = []
    for i, line in enumerate(line_interesting):
        if line[4] > minlen:
            j = i + 1