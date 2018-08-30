# coding=utf-8

from individual.__tree__ import DecisionTree
import networkx as nx
from io import StringIO
from matplotlib import pyplot as plt

__author__ = 'Henry Cagnini'


class Individual(DecisionTree):

    # relative tolerance between two accuracies
    rtol = 1e-3

    def __init__(self, gm):
        super(Individual, self).__init__(gm)

    def to_dot(self):
        output = StringIO()
        tree = self.tree  # type: nx.DiGraph
        nx.drawing.nx_pydot.write_dot(tree, output)
        _str = output.getvalue()
        output.close()
        return _str

    def plot(self, savepath=None):
        """
        Draw this individual.
        """

        from networkx.drawing.nx_agraph import graphviz_layout

        fig = plt.figure(figsize=(40, 30))

        tree = self.tree  # type: nx.DiGraph
        pos = graphviz_layout(tree, root=0, prog='dot')

        node_list = tree.nodes(data=True)
        edge_list = tree.edges(data=True)

        node_labels = {
            x[0]: '%s: %s\n%s' % (str(x[1]['node_id']), str(x[1]['label']), '%s/%s' % (
                str(x[1]['inst_correct']), str(x[1]['inst_total'])) if x[1]['terminal'] else '')
            for x in node_list
        }
        node_colors = [x[1]['color'] for x in node_list]
        edge_labels = {(x1, x2): d['threshold'] for x1, x2, d in edge_list}

        nx.draw_networkx_nodes(tree, pos, node_size=1000, node_color=node_colors)  # nodes
        nx.draw_networkx_edges(tree, pos, edgelist=edge_list, style='dashed')  # edges
        nx.draw_networkx_labels(tree, pos, node_labels, font_size=16)  # node labels
        nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels, font_size=16)

        plt.text(
            0.2,
            0.9,
            '\n'.join([
                'height: %d' % self.height,
                'n_nodes: %d' % self.n_nodes,
                'train accuracy: %0.4f' % self.train_acc_score,
                'val accuracy: %0.4f' % self.val_acc_score,
                'test accuracy: %0.4f' % self.test_acc_score if self.test_acc_score is not None else ''

            ]),
            fontsize=15,
            horizontalalignment='right',
            verticalalignment='top',
            transform=fig.transFigure
        )

        plt.axis('off')

        if savepath is not None:
            plt.savefig(savepath, bbox_inches='tight', format='pdf')
            plt.close()

    def __is_close__(self, other, attribute_name):
        quality_diff = abs(getattr(self, attribute_name) - getattr(other, attribute_name))
        return quality_diff <= Individual.rtol

    def __le__(self, other):  # less or equal
        if self.__is_close__(other, 'train_acc_score'):
            if self.height == other.height:
                return self.n_nodes >= other.n_nodes
            return self.height >= other.height
        return self.train_acc_score <= other.train_acc_score

    def __lt__(self, other):  # less than
        if self.__is_close__(other, 'train_acc_score'):
            if self.height == other.height:
                return self.n_nodes > other.n_nodes
            return self.height > other.height
        return self.train_acc_score < other.train_acc_score

    def __ge__(self, other):  # greater or equal
        if self.__is_close__(other, 'train_acc_score'):
            if self.height == other.height:
                return self.n_nodes <= other.n_nodes
            return self.height <= other.height
        return self.train_acc_score >= other.train_acc_score

    def __gt__(self, other):  # greater than
        if self.__is_close__(other, 'train_acc_score'):
            if self.height == other.height:
                return self.n_nodes < other.n_nodes
            return self.height < other.height
        return self.train_acc_score > other.train_acc_score

    def __eq__(self, other):  # equality
        if self.__is_close__(other, 'train_acc_score'):
            if self.height == other.height:
                return self.n_nodes == other.n_nodes
            return False
        return False

    def __ne__(self, other):  # inequality
        if self.__is_close__(other, 'train_acc_score'):
            if self.height == other.height:
                return not self.n_nodes == other.n_nodes
            return True
        return True

    def __str__(self):
        return 'train: %0.3f val: %0.3f n_nodes: %d height: %d' % (
            self.train_acc_score, self.val_acc_score, self.n_nodes, self.height
        )
