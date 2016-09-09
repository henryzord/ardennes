from plotly.offline import plot
from plotly.graph_objs import Bar, Scatter, Figure, Layout

# plot([
#     Scatter(x=[1, 2, 3], y=[3, 1, 6], name='some bars')
# ])

# saving figures, instead of plotting them
# Save the figure as a png image:
# py.image.save_as(fig, 'my_plot.png')


trace1 = Scatter(
    x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    y=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    name='Name of Trace 1'
)
trace2 = Scatter(
    x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    y=[1, 0, 3, 2, 5, 4, 7, 6, 8],
    name='Name of Trace 2'
)
data = [trace1, trace2]
layout = Layout(
    title='Plot Title',
    xaxis=dict(
        title='x Axis',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='y Axis',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
fig = Figure(data=data, layout=layout)
plot_url = plot(fig)  # , filename='styling-names')