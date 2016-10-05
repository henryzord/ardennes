# coding=utf-8
from plotly.offline import plot
from numpy import *

__author__ = 'Henry Cagnini'


def main(file_name):
    X = loadtxt(file_name, delimiter=',')

    N = 5

    # generate an array of rainbow colors by fixing the saturation and lightness of the HSL representation of colour
    # and marching around the hue.
    # Plotly accepts any CSS color format, see e.g. http://www.w3schools.com/cssref/css_colors_legal.asp.
    c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in linspace(0, 360, N)]

    # Each box is represented by a dict that contains the data, the type, and the colour.
    # Use list comprehension to describe N boxes, each with a different colour
    data = [{
                'y': X[i],
                'type': 'box',
                'name': 'generation #%01.d' % i,
                'marker': {'color': c[i]}
            } for i in xrange(N)]

    # format the layout
    layout = {'xaxis': {'showgrid': False, 'zeroline': False, 'tickangle': 60, 'showticklabels': False},
              'yaxis': {'zeroline': False, 'gridcolor': 'white'},
              'paper_bgcolor': 'rgb(233,233,233)',
              'plot_bgcolor': 'rgb(233,233,233)',
              }

    plot(data)


if __name__ == '__main__':
    filename = '/home/henry/Projects/forrestTemp/metadata/fold=00 run=00 10:05 05-10-2016.csv'
    main(filename)
