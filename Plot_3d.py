import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='otakulucy404', api_key='5y4acheGm5DgwjvuptMD')

import numpy as np
import pickle


x = pickle.load( open( "saveX.p", "rb" ) )
y = pickle.load( open( "saveY.p", "rb" ) )
z = pickle.load( open( "saveZ.p", "rb" ) )
trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
     marker=dict(
        size=12,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')