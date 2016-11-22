# coding=utf-8
import json
import os
import shutil

import arff
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

import warnings
warnings.filterwarnings('error')

__author__ = 'Henry Cagnini'


def generate_folds(df, dataset_name, output_folder, n_folds=10, random_state=None):
    """
    Given a dataset df, generate n_folds for it and store them into <output_folder>/<dataset_name>.json.

    :type df: pandas.DataFrame.
    :param df: The dataset, along class attribute.
    :type dataset_name: str
    :param dataset_name: Name of the dataset, for storing the results.
    :type output_folder: str
    :param output_folder: Directory to store the folds file.
    :type n_folds: int
    :param n_folds: Optional - Number of folds to split the dataset into. Defaults to 10.
    :type random_state: int
    :param random_state: Optional - Seed to use in the splitting process. Defaults to None (no seed).
    :return:
    """

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    _folds = skf.split(df[df.columns[:-1]], df[df.columns[-1]])

    d_folds = dict()

    for i, (arg_train, arg_test) in enumerate(_folds):
        x_train, x_val, y_train, y_val = train_test_split(
            df.iloc[arg_train][df.columns[:-1]],
            df.iloc[arg_train][df.columns[-1]],
            test_size=1. / (n_folds - 1.),
            random_state=random_state
        )

        d_folds[i] = {'train': list(x_train.index), 'val': list(x_val.index), 'test': list(arg_test)}
    json.dump(d_folds, open(os.path.join(output_folder, dataset_name + '.json'), 'w'), indent=2)


def read_dataset(dataset_path):
    dataset_type = dataset_path.split('.')[-1].strip()

    if dataset_type == 'csv':
        dataset = pd.read_csv(dataset_path, sep=',')
    elif dataset_type == 'arff':
        af = arff.load(open(dataset_path, 'r'))
        dataset = pd.DataFrame(af['data'], columns=[x[0] for x in af['attributes']])
    else:
        raise TypeError('Invalid type for dataset! Must be either \'csv\' or \'arff\'!')

    return dataset


def get_batch(dataset, train_size=0.8, random_state=None):
    train, rest = train_test_split(
        dataset,
        train_size=train_size,
        random_state=random_state
    )

    val, test = train_test_split(
        rest,
        test_size=0.5,  # validation and test set have the same proportion
        random_state=random_state
    )

    return train, val, test


def get_fold_iter(df, fold):
    """
    Given a dataset and a file with its folds, returns a iterator to iterate over such folds.

    :type df: pandas.DataFrame
    :param df: The full dataset.
    :param fold: Either a string that leads to a json file, or a json file itself.
    :return: A iterator which iterates over the folds, from the first to the last, return a tuple in the format
        (train_set, test_set, validation_set).
    """

    if isinstance(fold, str) or isinstance(fold, unicode):
        f = json.load(open(fold, 'r'))  # type: dict
    elif isinstance(fold, dict):
        f = fold  # type: dict
    else:
        raise TypeError('Invalid type for fold! Must be either a dictionary or a path to a json file!')

    for i in sorted(f.keys()):
        train_s = df.loc[f[i]['train']]
        test_s = df.loc[f[i]['test']]
        val_s = df.loc[f[i]['val']]

        yield train_s, test_s, val_s


def main():
    dataset_path = '../datasets/numerical'
    output_folder = '../datasets/folds'

    n_folds = 5

    for dataset_format in os.listdir(dataset_path):
        name = dataset_format.split('.')[0]
        df = read_dataset(os.path.join(dataset_path, dataset_format))

        print 'doing for dataset %s' % name
        # generate_folds(df, dataset_name=name, output_folder=output_folder, n_folds=n_folds)
        it = get_fold_iter(df, os.path.join(output_folder, name + '.json'))
        for train_s, test_s, val_s in it:
            z = 0


if __name__ == '__main__':
    main()
