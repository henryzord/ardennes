from __future__ import absolute_import

import sys
sys.path.append("..")

from c_individual import make_predictions

import numpy as np
import cPickle
from preprocessing.dataset import read_dataset
import itertools as it
from datetime import datetime as dt

df = read_dataset('/home/henry/Projects/ardennes/datasets/numerical/iris.arff')

some_ind = cPickle.load(open('individual.bin', 'r'))

# some_ind.plot()
# from matplotlib import pyplot as plt
# plt.show()
attribute_index = {k: x for x, k in enumerate(df.columns)}

data = df.values.ravel().tolist()

tree = some_ind.tree.node
t1 = dt.now()
offline_predictions = some_ind.predict(df)
t2 = dt.now()

online_predictions = make_predictions(
    df.shape,
    data,
    tree,
    range(df.shape[0]),
    attribute_index
)
t3 = dt.now()

for x, y in it.izip(offline_predictions, online_predictions):
    print 'off: ', x, ' on: ', y

print 'off: %f on: %f' % ((t2 - t1).total_seconds(), (t3 - t2).total_seconds())
