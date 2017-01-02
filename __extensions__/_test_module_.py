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

individual = cPickle.load(open('/home/henry/Desktop/individual', 'r'))
make_predictions(df.shape[0], df.shape[1])
