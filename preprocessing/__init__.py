import os
import pandas as pd
from sklearn.model_selection import train_test_split

__author__ = 'Henry Cagnini'


def __split__(data, train_size):
    """
    Splits data into two subsets.

    :param data: Data to be split into two dataset.
    :type train_size: float
    :param train_size: The size of the first set to be returned.
    :rtype: tuple
    :return: a tuple where the first element is the training set and the other set.
    """

    assert isinstance(train_size, float), TypeError('train_size must be a float with the size of the first set!')
    assert 0 < train_size <= 1, ValueError('train_size must be within (0, 1]!')

    if train_size == 1.:
        return data, None

    train_s, val_s = train_test_split(data, train_size=train_size)

    return train_s, val_s


def get_dataset_name(dataset_path):
    if os.name == 'nt':
        sep = '\\'
    else:
        sep = '/'

    return dataset_path.split(sep)[-1].split('.')[0]