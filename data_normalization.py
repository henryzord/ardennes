"""
You must override the class in this script for defining a data normalization strategy.
"""

__author__ = 'Henry Cagnini'


class DataNormalizer(object):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        return X
