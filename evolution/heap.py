import numpy as np


__author__ = 'Henry Cagnini'


class Node(object):
    """
    Convenient class for heap tree operations and properties.
    """
    
    _root = 0
    
    @property
    def root(self):
        return self._root
    
    @staticmethod
    def get_left_child(id_node):
        return (id_node * 2) + 1
    
    @staticmethod
    def get_right_child(id_node):
        return (id_node * 2) + 2
    
    @staticmethod
    def get_parent(id_node):
        if id_node > 0:
            return int((id_node - 1) / 2.)
        return None
    
    @staticmethod
    def get_depth(id_node):
        """
        Gets depth of node in a binary heap.
        :param id_node: ID of the node in the binary heap.
        :return: The depth of the node.
        """
        return int(np.log2(id_node + 1))
    
    @staticmethod
    def get_node_count(n_nodes):
        """
        Given the total number of nodes in a binary tree, returns the number of internal and leaf nodes.
        :param n_nodes: Total number of nodes in a binary tree.
        :return: A tuple where the first item is the number of internal nodes, and the second the number of leaf nodes.
        """
        n_leaf = (n_nodes + 1) / 2
        n_internal = n_nodes - n_leaf
        return n_internal, n_leaf
