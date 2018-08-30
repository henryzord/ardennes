"""
File for support functions.
"""

import os

import arff
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def path_to_arff(dataset_path):
    """
    Given a path to a dataset, reads and returns a dictionary which comprises an arff file.

    :type dataset_path: str
    :param dataset_path: Path to the dataset. Must contain the .arff file extension (i.e., "my_dataset.arff")
    :rtype: dict
    :return: a dictionary with the arff dataset.
    """

    dataset_type = dataset_path.split('.')[-1].strip()
    assert dataset_type == 'arff', TypeError('Invalid type for dataset! Must be an \'arff\' file!')
    af = arff.load(open(dataset_path, 'r'))
    return af


def path_to_dataframe(dataset_path):
    """
    Given a path to an arff dataset, transforms it to a pandas.DataFrame.

    :type dataset_path: str
    :param dataset_path: Path to the dataset. Must contain the .arff file extension (i.e., "my_dataset.arff")
    :return: the dataset as a dataframe.
    :rtype: pandas.DataFrame
    """

    file_arff = path_to_arff(dataset_path)

    file_df = pd.DataFrame(
        data=file_arff['data'],
        columns=[x[0] for x in file_arff['attributes']],
    )

    file_df.replace('?', np.nan, inplace=True)

    # for column in file_df.columns[:-1]:  # until last attribute
    #     file_df[column] = pd.to_numeric(file_df[column])
    #     file_df[column].fillna(file_df[column].mean(), inplace=True)

    return file_df


def __get_fold__(params, dataset_path, n_fold):
    """
    Returns the full dataset, along indices for each set.

    If performing an e.g. 10-fold cross-validation, the training set has 9 folds, and the testing set 1 fold.

    :type params: dict
    :param params: The parameter file as a dictionary of values.
    :type dataset_path: str
    :param dataset_path: a full path (including file type, i.e., dataset.arff) to the dataset to be run.
    :type n_fold: int
    :param n_fold: fold to get for testing. Must be smaller than params['n_folds'] (e.g., if params['n_folds'] = 3,
        then the fold indices are [0, 1, 2]).
    :rtype: tuple
    :return: whole_dataset, X_train_indices, X_test_indices based on the provided n_fold.
    """

    assert n_fold < params['n_folds'], ValueError('n_fold must be less than total number of folds in params file!')

    skf = StratifiedKFold(n_splits=params['n_folds'], shuffle=True, random_state=params['random_state'])

    full_df = path_to_dataframe(dataset_path)

    y_name = full_df.columns[-1]

    full_df[y_name] = pd.Categorical(full_df[y_name])
    # full_df[y_name] = full_df[y_name].cat.codes

    X = full_df[full_df.columns[:-1]]
    y = full_df[full_df.columns[-1]]

    train_index = None
    test_index = None
    for fold_num, (train_index_, test_index_) in enumerate(skf.split(X, y)):
        if fold_num == n_fold:
            train_index = train_index_
            test_index = test_index_
            break

    return full_df, train_index, test_index


def get_dataset_name(dataset_path):
    """
    Given a path, gets the dataset name.
    :type dataset_path: str
    :param dataset_path: A path to a dataset. May (or may not) contain file extension and be within a folder.
    :rtype: str
    :return: the name of the dataset.
    """

    return dataset_path.split(os.sep)[-1].split('.')[-2]


class MetaDataset(object):
    def __init__(self, dataset):
        self.n_objects, self.n_attributes = dataset.shape

        self.pred_attr = np.array(dataset.columns[:-1])  # type: np.ndarray
        self.target_attr = str(dataset.columns[-1])  # type: str
        self.class_labels = np.sort(dataset[dataset.columns[-1]].unique())  # type: np.ndarray

        self.numerical_class_labels = np.arange(len(self.class_labels), dtype=np.int32)  # type: np.ndarray
        self.class_label_index = {k: x for x, k in enumerate(self.class_labels)}  # type: dict
        self.inv_class_label_index = {x: k for x, k in enumerate(self.class_labels)}  # type: dict
        self.attribute_index = {k: x for x, k in enumerate(dataset.columns)}  # type: dict

        self.column_types = {x: self.raw_type_dict[str(dataset[x].dtype)] for x in dataset.columns}  # type: dict

    def to_onehot(self, y):
        """
        Adapted from https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py#L10
        Converts a class vector (integers) to binary class matrix.
        E.g. for use with categorical_crossentropy.
        # Arguments
            y: class vector to be converted into a matrix
                (integers from 0 to nb_classes).
            nb_classes: total number of classes.
        # Returns
            A binary matrix representation of the input.
        """
        nb_classes = self.numerical_class_labels.shape[0]

        y = np.array(y, dtype='int').ravel()
        if not nb_classes:
            nb_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, nb_classes))
        categorical[np.arange(n), y] = 1
        return categorical

    def get_predictive_type(self, dtype):
        """
        Tells whether the attribute is categorical or numerical.

        :type dtype: type
        :param dtype: dtype of an attribute.
        :rtype: str
        :return: Whether this attribute is categorical or numerical.
        """

        raw_type = self.raw_type_dict[str(dtype)]
        mid = self.mid_type_dict[raw_type]
        return mid

    arff_type_dict = {
        'numeric': 'float',
        'real': 'float',
        'continuous': 'float',
    }

    raw_type_dict = {
        'int': 'int',
        'int_': 'int',
        'intc': 'int',
        'intp': 'int',
        'int8': 'int',
        'int16': 'int',
        'int32': 'int',
        'int64': 'int',
        'uint8': 'int',
        'uint16': 'int',
        'uint32': 'int',
        'uint64': 'int',
        'float': 'float',
        'float_': 'float',
        'float16': 'float',
        'float32': 'float',
        'float64': 'float',
        'complex_': 'complex',
        'complex64': 'complex',
        'complex128': 'complex',
        'object': 'object',
        'bool_': 'bool',
        'bool': 'bool',
        'str': 'str',
        'category': 'category'
    }

    mid_type_dict = {
        'object': 'categorical',
        'str': 'categorical',
        'int': 'numerical',
        'float': 'numerical',
        'bool': 'categorical',
        'category': 'categorical',
    }

    arff_data_types = {
        'date': 'numerical',
        'string': 'categorical',
        'integer': 'numerical',
        'numeric': 'numerical',
        'continuous': 'numerical',
        'real': 'numerical'
    }

    numerical = 'numerical'
    categorical = 'categorical'
