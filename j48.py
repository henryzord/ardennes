import pandas as pd
import numpy as np
from preprocessing.dataset import read_dataset
from matplotlib import pyplot as plt

from parallel import Handler
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from itertools import count


def plot(tree):
    """
    Draw this individual.
    """

    # from wand.image import Image
    # from wand import display
    # img = Image(filename='.temp.pdf')
    # display.display(img)

    # fig = plt.figure(figsize=(40, 30))

    pos = graphviz_layout(tree, root=0, prog='dot')

    node_list = tree.nodes(data=True)
    edge_list = tree.edges(data=True)

    node_labels = {x[0]: str(x[1]['node_id']) + ': ' + str(x[1]['label']) for x in node_list}
    edge_labels = {(x1, x2): d['threshold'] for x1, x2, d in edge_list}

    nx.draw_networkx_nodes(tree, pos, node_size=1000, node_color='#FFFFFF')  # nodes
    nx.draw_networkx_edges(tree, pos, edgelist=edge_list, style='dashed')  # edges
    nx.draw_networkx_labels(tree, pos, node_labels, font_size=16)  # node labels
    nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels, font_size=16)

    plt.axis('off')
    plt.show()


def hunt(subset, tree, parent, handler, n_instances):
    node_id = next(counter)

    if len(subset[subset.columns[-1]].unique()) == 1:
        tree.add_node(node_id,
                      attr_dict={'label': subset[subset.columns[-1]].unique(), 'node_id': node_id}
                      )
        tree.add_edge(parent, node_id, attr_dict={'threshold': tree.node[parent]['threshold']})
        return tree

    if subset.shape[0] == 1:

        tree.add_node(node_id,
                      attr_dict={'label': subset.loc[subset.index[0], subset.columns[-1]], 'node_id': node_id}
                      )
        tree.add_edge(parent, node_id, attr_dict={'threshold': tree.node[parent]['threshold']})
        return tree

    best_gain = -np.inf
    best_threshold = None
    best_attribute = None
    for attribute in subset.columns[:-1]:
        unique_vals = [float(x) for x in sorted(subset[attribute].unique())]
        # candidates = unique_vals[1:-1]

        candidates = np.array(unique_vals + [
            (unique_vals[i] + unique_vals[i + 1]) / 2.
            if (i + 1) < len(unique_vals) else unique_vals[i] for i in
            xrange(len(unique_vals))
        ][:-1])
        # candidates.sort()
        # print 'candidates:', candidates

        subset_index = np.zeros(n_instances)
        subset_index[subset.index] = 1

        gains = handler.batch_gain_ratio(subset_index, attribute=attribute, candidates=candidates)

        argmax = np.argmax(gains)
        if gains[argmax] > best_gain:
            best_candidates = candidates
            best_gains = gains
            best_gain = gains[argmax]
            best_threshold = candidates[argmax]
            best_attribute = attribute

    tree.add_node(node_id, attr_dict={'label': best_attribute, 'threshold': best_threshold, 'node_id': node_id})
    if parent is not None:
        tree.add_edge(parent, node_id, attr_dict={'threshold': tree.node[parent]['threshold']})

    subset_left, subset_right = subset.loc[subset[best_attribute] <= best_threshold], \
                                subset.loc[subset[best_attribute] > best_threshold]

    print 'subset: %d left: %d right: %d' % (subset.shape[0], subset_left.shape[0], subset_right.shape[0])

    if subset_left.shape[0] <= 0 or subset_right.shape[0] <= 0:
        tree.add_node(node_id,
                      attr_dict={'label': subset.loc[subset.index[0], subset.columns[-1]], 'node_id': node_id}
                      )
        return tree

    tree = hunt(subset_left, tree, node_id, handler, n_instances)
    tree = hunt(subset_right, tree, node_id, handler, n_instances)
    return tree


def main():
    _dataset = read_dataset('datasets/numerical/hayes-roth-full.arff')
    n_instances = _dataset.shape[0]
    _handler = Handler(_dataset)
    _tree = nx.DiGraph()
    _tree = hunt(_dataset, _tree, None, _handler, n_instances)
    plot(_tree)

if __name__ == '__main__':
    counter = count(0, 1)
    main()
