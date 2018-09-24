from devices import Device
from utils import path_to_dataframe
from cpu_device import gain_ratio
import numpy as np
import pandas as pd

from datetime import datetime as dt

full_df = path_to_dataframe('../datasets/iris.arff')

column_number = 0
column_name = full_df.columns[column_number]
n_classes = len(full_df[full_df.columns[-1]].unique())

candidates = np.linspace(np.min(full_df[column_name].min()), np.min(full_df[column_name].max()), num=5, dtype=np.float32)

full_df[full_df.columns[-1]] = pd.Categorical(full_df[full_df.columns[-1]])
full_df[full_df.columns[-1]] = full_df[full_df.columns[-1]].cat.codes  # TODO change later!

subset_index = np.ones(len(full_df), dtype=np.bool)

for column in full_df.columns:
    full_df[column] = full_df[column].astype(np.float32)

dataset = full_df.values

t1 = dt.now()
subset_ratios = gain_ratio(dataset, subset_index, 0, candidates, n_classes)
t2 = dt.now()
correct_ratios = Device.get_gain_ratios(full_df, subset_index, column_name, candidates)
t3 = dt.now()

print('C++: %f\t Python: %f\n' % ((t2 - t1).total_seconds(), (t3 - t2).total_seconds()))

print('Python gain ratios:', correct_ratios)
print('   C++ gain ratios:', subset_ratios)
