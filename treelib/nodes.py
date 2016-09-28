# coding=utf-8

__author__ = 'Henry Cagnini'


class Node(object):
    _root = 0

    def __init__(self, arg=None, label=None, threshold=None, parent=None, left=None, right=None):
        self.arg = arg
        self.label = label
        self.threshold = threshold
        self.left = left
        self.right = right
        self.parent = parent

    @property
    def is_terminal(self):
        return self.threshold is None

    @property
    def is_internal(self):
        return not self.is_terminal()

    @property
    def is_root(self):
        return self.arg == self.__class__._root
