import numpy as np
from skimage.measure import LineModelND, ransac
from PlaneModelC import PlaneModelND


idx_row = np.random.randint(200, size=(10, 1))
idx_col = np.random.randint(200, size=(10, 1))

pointcloud = np.random.randint(500, size=(480, 640, 3))
pointcloud[:, :, 2] = 0

data1 = pointcloud[idx_row, idx_col, :]
data1 = np.squeeze(data1)
# print(data1.shape)
print(data1)

# robustly fit plane and line only using inlier data with RANSAC algorithm
model_robustP, inliersP = ransac(data1, PlaneModelND,  min_samples=3, residual_threshold=1, max_trials=20)
print(inliersP)
print(model_robustP.params)

new_x = []
new_y = []
new_z = []
for i in range(len(data1)):
    if inliersP[i] == True:
        new_x.append(data1[i][0])
        new_y.append(data1[i][1])
        new_z.append(data1[i][2])

np.save("new_x.npy", new_x)
np.save("new_y.npy", new_y)
np.save("new_z.npy", new_z)

print(new_x)
