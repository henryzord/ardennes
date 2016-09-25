from plotly.offline import plot
import plotly.graph_objs as go

trace1 = go.Bar(
    y=['giraffes', 'orangutans', 'monkeys'],
    x=[20, 14, 23],
    name='SF Zoo',
    orientation='h',
    marker=dict(
        color='rgba(246, 78, 139, 0.6)',
        line=dict(
            color='rgba(246, 78, 139, 1.0)',
            width=3
        )
    )
)

trace2 = go.Bar(
    y=['giraffes', 'orangutans', 'monkeys'],
    x=[12, 18, 29],
    name='LA Zoo',
    orientation='h',
    marker=dict(
        color='rgba(58, 71, 80, 0.6)',
        line=dict(
            color='rgba(58, 71, 80, 1.0)',
            width=3)
    )
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
plot(fig, filename='marker-h-bar')
