"""
Convenient module for heap (i.e, binary) tree properties.
"""
# coding=utf-8

import numpy as np
from math import ceil, floor


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


def get_node_count(n_nodes):
    """
    Given the total number of nodes in a binary tree, returns the number of internal and leaf nodes.

    :param n_nodes: Total number of nodes in a binary tree.
    :return: A tuple where the first item is the number of internal nodes, and the second the number of leaf nodes.
    """
    n_internal = n_nodes - n_leaf
    n_leaf = (n_nodes + 1) / 2
    return n_internal, n_leaf


def total_nodes_by_height(height):
    """
    Get number of total nodes from a tree with given height.

    :type height: int
    :param height: Number of levels in the binary tree.
    :rtype: int
    :return: The number of nodes in this tree.
    """
    return np.power(2, height) - 1


def nodes_at_level(level):
    """
    Get number of nodes at given level.

    :type level: int
    :param level: The querying level.
    :rtype: int
    :return: Number of nodes in this level.
    """

    return int(np.power(2, level))


def index_at_level(level):
    """
    Picks the index of all the nodes at a given level in a binary tree.

    :type level: int
    :param level: The level, starting at zero.
    :rtype: list of int
    :return: A list of the indices of nodes at a given level.
    """
    before = nodes_at_level(level - 1)
    current = nodes_at_level(level)

    index = range(before, before + current)
    return index


def get_max_height(n_nodes):
    """
    Get the height of the tree by supplying the number of nodes in it.

    :param n_nodes: Total number of nodes in a binary tree.
    :return: Minimum height of a tree with the given number of nodes.
    """
    h = round((n_nodes - 1.) / 2.)
    return h


def get_min_height(n_nodes):
    """
    Get the height of the tree by supplying the number of nodes in it.

    :param n_nodes: Total number of nodes in a binary tree.
    :return: Maximum height of a tree with the given number of nodes.
    """
    h = __closest_power__(n_nodes) - 1
    return h


def __closest_power__(n):
    log = np.log2(n)
    upper_diff = abs(2.**log - (2. ** ceil(log)))
    lower_diff = abs(2.**log - (2. ** floor(log)))

    power = (ceil(log) * (upper_diff < lower_diff)) + (floor(log) * (lower_diff <= upper_diff))

    return power

if __name__ == '__main__':
    __closest_power__(255)
