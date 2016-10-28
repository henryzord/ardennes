from cStringIO import StringIO

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx

# TODO needs to be upgraded in order to run:
# TODO please take a look at
# http://networkx.readthedocs.io/en/latest/reference/migration_guide_from_1.x_to_2.0.html

# create a networkx graph
G = nx.MultiDiGraph()
G.add_nodes_from([1, 2])
G.add_edge(1, 2)

# convert from networkx -> pydot
pydot_graph = nx.nx_pydot.to_pydot(G)

# render pydot by calling dot, no file saved to disk
png_str = pydot_graph.create_png(prog='dot')

# treat the dot output string as an image file
sio = StringIO()
sio.write(png_str)
sio.seek(0)
img = mpimg.imread(sio)

# plot the image
imgplot = plt.imshow(img, aspect='equal')
plt.show()
