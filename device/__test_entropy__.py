from device import Device
from utils import path_to_dataframe
from cpu_device import gain_ratio
import numpy as np
import pandas as pd

full_df = path_to_dataframe('../datasets/iris.arff')
correct_entropy = Device.entropy(full_df)

candidates = np.array([np.min(full_df[full_df.columns[0]]), np.max(full_df[full_df.columns[0]])], dtype=np.float32)
n_classes = len(full_df[full_df.columns[-1]].unique())

full_df[full_df.columns[-1]] = pd.Categorical(full_df[full_df.columns[-1]])
full_df[full_df.columns[-1]] = full_df[full_df.columns[-1]].cat.codes  # TODO change later!

subset_index = np.ones(len(full_df), dtype=np.bool)

dataset = np.ascontiguousarray(full_df.values)

subset_entropy = gain_ratio(dataset, subset_index, 0, candidates, n_classes)
print('correct entropy:', correct_entropy)
print("subset entropy:", subset_entropy)