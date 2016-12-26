import os
import numpy as np


class Device(object):
    def __init__(self, dataset):
        self.n_objects, self.n_attributes = dataset.shape

        self.target_attr = dataset.columns[-1]
        self.class_labels = np.array(dataset[dataset.columns[-1]].unique())
        self.numerical_class_labels = np.arange(self.class_labels.shape[0], dtype=np.float32)
        self.class_label_index = {k: x for x, k in enumerate(self.class_labels)}
        self.attribute_index = {k: x for x, k in enumerate(dataset.columns)}

        sep = '\\' if os.name == 'nt' else '/'

        cur_path = os.path.abspath(__file__)
        self._split = sep.join(cur_path.split(sep)[:-1])

    def queue_execution(self, subset_index, attribute, candidates):
        pass
