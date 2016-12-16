
from base import Handler
import numpy as np


class SequentialHandler(Handler):
    def batch_gain_ratio(self, subset_index, attribute, candidates):
        proper_subset = self.dataset.loc[subset_index.astype(np.bool)]

        ratios = map(
            lambda c: self.gain_ratio(
                proper_subset,
                proper_subset.loc[proper_subset[attribute] < c],
                proper_subset.loc[proper_subset[attribute] >= c],
                self.target_attr
            ),
            candidates
        )
        return ratios
