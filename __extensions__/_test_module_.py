from __future__ import absolute_import

import sys
sys.path.append("..")
from treelib import individual

from c_individual import make_predictions

from sklearn import datasets
import pandas as pd
import numpy as np
import cPickle

dt = datasets.load_iris()
df = pd.DataFrame(
    data=np.hstack((dt.data.astype(np.float32), dt.target[:, np.newaxis].astype(np.float32))),
    columns=np.hstack((dt.feature_names, 'class'))
)

some_ind = cPickle.load(open('individual.bin', 'r'))

tree = some_ind.tree.node

print make_predictions(
    df.shape[0], df.shape[1], df.values.ravel().astype(np.float32).tolist(), tree, (np.ones(df.shape[0], dtype=np.int32) * 42).tolist()
)
