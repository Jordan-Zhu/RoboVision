import numpy as np
from itertools import groupby

# my area boundary box
xmax, xmin, ymax, ymin = 640000.06, 636999.83, 6070000.3, 6066999.86

# generate fake data
ndata = 50
# Generate random data to simulate
x = np.random.randint(xmin, xmax, ndata)
y = np.random.randint(ymin, ymax, ndata)
z = np.random.randint(0,20,ndata)
mypoints = zip(x,y,z)
print(*mypoints)

keyfunc = lambda p: p[:2]
mypoints = []
for k, g in groupby(sorted(zip(x, y, z), key=keyfunc), keyfunc):
    mypoints.append(list(g)[0])

print(mypoints)