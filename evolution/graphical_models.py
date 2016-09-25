from heap import Node

import numpy as np
import networkx as nx
import pandas as pd
from treelib.setter import SetterClass


__author__ = 'Henry Cagnini'


class AbstractGraphicalModel(SetterClass):
    pred_attr = None
    target_attr = None
    class_labels = None
    
    def __init__(self, pred_attr, target_attr, class_labels):
        self.__class__.set_values(
            pred_attr=pred_attr,
            target_attr=target_attr,
            class_labels=class_labels
        )

    def plot(self):
        pass


class GraphicalModel(AbstractGraphicalModel):
    """
        A graphical model is a tree itself.
    """

    tensor = None  # tensor is a dependency graph

    def __init__(self, pred_attr, target_attr, class_labels, pattern=None):
        super(GraphicalModel, self).__init__(pred_attr, target_attr, class_labels)

        self.tensor = None