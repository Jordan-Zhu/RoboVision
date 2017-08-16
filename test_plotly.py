import plotly.plotly as py
import plotly.graph_objs as go
import plotly

import numpy as np
from sklearn import linear_model, datasets
plotly.tools.set_credentials_file(username='jzhu1', api_key='dHJj9eRDVTfTiOYZmtbf')


n_samples = 1000
n_outliers = 50


X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
                                      n_informative=1, noise=10,
                                      coef=True, random_state=0)

# Add outlier data
np.random.seed(0)
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

# Fit line using all data
model = linear_model.LinearRegression()
model.fit(X, y)

# Robustly fit linear model with RANSAC algorithm
model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
model_ransac.fit(X, y)
inlier_mask = model_ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
line_X = np.arange(-5, 5)
line_y = model.predict(line_X[:, np.newaxis])
line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])

# Compare estimated coefficients
print("Estimated coefficients (true, normal, RANSAC):")
print(coef, model.coef_, model_ransac.estimator_.coef_)

# Plot results
def data_to_plotly(x):
    k = []

    for i in range(0, len(x)):
        k.append(x[i][0])

    return k

lw = 2

p1 = go.Scatter(x=data_to_plotly(X[inlier_mask]), y=y[inlier_mask],
                mode='markers',
                marker=dict(color='yellowgreen', size=6),
                name='Inliers')
p2 = go.Scatter(x=data_to_plotly(X[outlier_mask]), y=y[outlier_mask],
                mode='markers',
                marker=dict(color='gold', size=6),
                name='Outliers')

p3 = go.Scatter(x=line_X, y=line_y,
                mode='lines',
                line=dict(color='navy', width=lw,),
                name='Linear regressor')
p4 = go.Scatter(x=line_X, y=line_y_ransac,
                mode='lines',
                line=dict(color='cornflowerblue', width=lw),
                name='RANSAC regressor')
data = [p1, p2, p3, p4]
layout = go.Layout(xaxis=dict(zeroline=False, showgrid=False),
                   yaxis=dict(zeroline=False, showgrid=False)
                  )
fig = go.Figure(data=data, layout=layout)

py.iplot(fig)