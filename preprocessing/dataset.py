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
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    _folds = skf.split(df[df.columns[:-1]], df[df.columns[-1]])

    write_path = os.path.join(output_folder, dataset_name)

    if os.path.exists(write_path):
        shutil.rmtree(write_path)
    os.mkdir(write_path)

    for i, (arg_train, arg_test) in enumerate(_folds):
        x_train, x_val, y_train, y_val = train_test_split(
            df.iloc[arg_train][df.columns[:-1]],
            df.iloc[arg_train][df.columns[-1]],
            test_size=1. / (n_folds - 1.),
            random_state=random_state
        )

        _dict = {'train': list(x_train.index), 'val': list(x_val.index), 'test': list(arg_test)}
        json.dump(_dict, open(os.path.join(write_path, dataset_name + '_fold_%03.d' % i + '.json'), 'w'), indent=2)

    return _folds


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


def fold_index_to_data(fold_index):
    pass


def main():
    dataset_path = '../datasets/numerical'
    output_folder = '../datasets/folds'

    n_folds = 5

    for dataset_format in os.listdir(dataset_path):
        name = dataset_format.split('.')[0]

        print 'doing for dataset %s' % name

        df = read_dataset(os.path.join(dataset_path, dataset_format))
        generate_folds(df, dataset_name=name, output_folder=output_folder, n_folds=n_folds)


if __name__ == '__main__':
    main()
