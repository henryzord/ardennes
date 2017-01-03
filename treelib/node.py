# coding=utf-8

"""
Convenient module for heap (i.e, binary) tree properties.
"""

import numpy as np


__author__ = 'Henry Cagnini'

root = 0


def get_left_child(id_node):
    return (id_node * 2) + 1


def get_right_child(id_node):
    return (id_node * 2) + 2


def get_parent(id_node):
    if id_node > 0:
        return int((id_node - 1) / 2.)
    return None


def get_depth(id_node):
    """
    Gets depth of node in a binary heap.
    :param id_node: ID of the node in the binary heap.
    :return: The depth of the node.
    """
    return int(np.log2(id_node + 1))


def get_total_nodes(max_depth):
    """
    Get number of total nodes from a tree with given depth.

    :type depth: int
    :param depth: The depth of the heap (binary) tree. Starts with zero (i.e, only root).
    :rtype: int
    :return: The number of nodes in this tree.
    """
    return np.power(2, max_depth + 1) - 1


def nodes_at_level(level):
    """
    Get number of nodes at given level.

    :type level: int
    :param level: The querying level.
    :rtype: int
    :return: Number of nodes in this level.
    """

    return np.power(2, level)
