# Line object with properties:
# start, end, object/background, curvature/discontinuity
# distance from another line, check if line is overlapping


class Line(object):
    start = [0, 0]  # (x, y)
    end = [0, 0]
    index = 0
    relation = "positive"   # object is on the positive/negative side of the line
    type = "curvature"    # curvature if it's on the object, discontinuity of there is nothing beyond it

    # The class "constructor" - It's actually an initializer
    def __init__(self, start, end, index, relation, type):
        self.start = start
        self.end = end
        self.index = index
        self.relation = relation
        self.type = type


# This does the same thing as Line.__init__
# def make_line(start, end):
#     line = Line(start, end)
#     return line
