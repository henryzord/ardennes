# coding=utf-8

import json
import os

import arff
import pandas as pd
import weka.core.jvm as jvm
from sklearn.model_selection import StratifiedKFold
from weka.core.converters import Loader, Saver

from treelib import MetaDataset

__author__ = 'Henry Cagnini'


def load_arff(dataset_path):
    """
    Given a path to a dataset, reads and returns a dictionary which comprises an arff file.

    :type dataset_path: str
    :param dataset_path: Path to the dataset. Must contain the .arff file extension (i.e "my_dataset.arff")
    :rtype: dict
    :return: a dictionary with the arff dataset.
    """

    dataset_type = dataset_path.split('.')[-1].strip()
    assert dataset_type == 'arff', TypeError('Invalid type for dataset! Must be an \'arff\' file!')
    af = arff.load(open(dataset_path, 'r'))
    return af


def load_dataframe(af):
    """

    :type af: dict
    :param af: Arff dataset.
    :rtype: pandas.DataFrame
    :return: a DataFrame with the dataset.
    """

    assert isinstance(af, dict), TypeError('You must pass a dictionary comprising an arff dataset to this function!')
    import numpy as np

    df = pd.DataFrame(
        data=af['data'],
        columns=[x[0] for x in af['attributes']]
    )
    # df = df.replace('?', df.replace(['?'], [None]))  # replaces missing data with None
    df = df.replace('?', np.nan)

    for attr, dtype in af['attributes']:
        if isinstance(dtype, list):
            pass
        elif MetaDataset.arff_data_types[dtype.lower()] == MetaDataset.numerical.lower():
            df[attr] = pd.to_numeric(df[attr])
        elif dtype == MetaDataset.categorical:
            df[attr].dtype = np.object

    return df


def generate_folds(dataset_path, output_folder, n_folds=10, random_state=None):
    """
    Given a dataset df, generate n_folds for it and store them in <output_folder>/<dataset_name>.

    :type dataset_path: str
    :param dataset_path: Path to dataset with .arff file extension (i.e my_dataset.arff)
    :type output_folder: str
    :param output_folder: Path to store both index file with folds and fold files.
    :type n_folds: int
    :param n_folds: Optional - Number of folds to split the dataset into. Defaults to 10.
    :type random_state: int
    :param random_state: Optional - Seed to use in the splitting process. Defaults to None (no seed).
    """

    import warnings
    warnings.filterwarnings('error')

    dataset_name = dataset_path.split('/')[-1].split('.')[0]

    af = load_arff(dataset_path)
    df = load_dataframe(af)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_iter = skf.split(df[df.columns[:-1]], df[df.columns[-1]])

    fold_index = dict()

    jvm.start()

    csv_loader = Loader(classname="weka.core.converters.CSVLoader")
    arff_saver = Saver(classname='weka.core.converters.ArffSaver')

    for i, (arg_rest, arg_test) in enumerate(fold_iter):
        fold_index[i] = list(arg_test)

        _temp_path = 'temp_%s_%d.csv' % (dataset_name, i)

        fold_data = df.loc[arg_test]  # type: pd.DataFrame
        fold_data.to_csv(_temp_path, sep=',', index=False)

        java_arff_dataset = csv_loader.load_file(_temp_path)
        java_arff_dataset.relationname = af['relation']
        java_arff_dataset.class_is_last()
        arff_saver.save_file(java_arff_dataset, os.path.join(output_folder, '%s_fold_%d.arff' % (dataset_name, i)))

        os.remove(_temp_path)

    json.dump(
        fold_index, open(os.path.join(output_folder, dataset_name + '.json'), 'w'), indent=2
    )

    jvm.stop()
    warnings.filterwarnings('default')


# def get_batch(dataset, train_size=0.8, random_state=None):
#     train, rest = train_test_split(
#         dataset,
#         train_size=train_size,
#         random_state=random_state
#     )
#
#     val, test = train_test_split(
#         rest,
#         test_size=0.5,  # validation and test set have the same proportion
#         random_state=random_state
#     )
#
#     return train, val, test

def get_folds_index(dataset_name, fold_path):
    fold_file = json.load(open(os.path.join(fold_path, dataset_name + '.json'), 'r'))  # type: dict
    return fold_file


def get_folds_data(full, dataset_name, fold_path):
    """

    :type full: pandas.DataFrame
    :param full: A DataFrame which comprises the whole dataset.
    :type dataset_name: str
    :param dataset_name: The name of the dataset.
    :type fold_path: str
    :param fold_path: Path to fold files.
    :rtype: dict
    :return: a dictionary with fold keys and data.
    """

    fold_file = json.load(open(os.path.join(fold_path, dataset_name + '.json'), 'r'))  # type: dict

    fold_data = dict()

    for n_fold, fold_index in fold_file.iteritems():
        fold_data[n_fold] = full.loc[fold_index]

    return fold_data


def get_dataset_sets(dataset_name, datasets_path, fold_file, output_path='.', merge_val=True):

    raise NotImplementedError('not implemented yet!')

    arff_dtst = arff.load(open(os.path.join(datasets_path, dataset_name + '.arff'), 'r'))

    macros = dict()

    for n_fold, folds_sets in fold_file.iteritems():
        n_fold = int(n_fold)
        macros[n_fold] = dict()

        macro_train = os.path.join(output_path, '%s_fold_%d_train.csv') % (dataset_name, int(n_fold))
        macro_test = os.path.join(output_path, '%s_fold_%d_test.csv') % (dataset_name, int(n_fold))

        attributes = [x[0] for x in arff_dtst['attributes']]
        np_train_s = pd.DataFrame(arff_dtst['data'], columns=attributes)
        np_test_s = pd.DataFrame(arff_dtst['data'], columns=attributes)

        np_test_s = np_test_s.loc[folds_sets['test']]  # type: pd.DataFrame
        if merge_val:
            np_train_s = np_train_s.loc[folds_sets['train'] + folds_sets['val']]  # type: pd.DataFrame
        else:
            macro_val = os.path.join(output_path, '%s_fold_%d_val.csv') % (dataset_name, int(n_fold))
            np_train_s = np_train_s.loc[folds_sets['train']]
            np_val_s = pd.DataFrame(arff_dtst['data'], columns=attributes).loc[folds_sets['val']]
            np_val_s = np_val_s.sort_values(by=np_val_s.columns[-1])
            np_val_s.to_csv(macro_val, index=False)
            macros[n_fold]['val'] = macro_val

        np_train_s = np_train_s.sort_values(by=np_train_s.columns[-1])
        np_test_s = np_test_s.sort_values(by=np_test_s.columns[-1])

        np_train_s.to_csv(macro_train, index=False)
        np_test_s.to_csv(macro_test, index=False)

        macros[n_fold]['train'] = macro_train
        macros[n_fold]['test'] = macro_test

    return macros


raw_type_dict = {
        'int': 'NUMERIC',
        'int_': 'NUMERIC',
        'intc': 'NUMERIC',
        'intp': 'NUMERIC',
        'int8': 'NUMERIC',
        'int16': 'NUMERIC',
        'int32': 'NUMERIC',
        'int64': 'NUMERIC',
        'uint8': 'NUMERIC',
        'uint16': 'NUMERIC',
        'uint32': 'NUMERIC',
        'uint64': 'NUMERIC',
        'float': 'REAL',
        'float_': 'REAL',
        'float16': 'REAL',
        'float32': 'REAL',
        'float64': 'REAL',
        'object': '{%s}',
        'bool_': 'NUMERIC',
        'bool': 'NUMERIC',
        'str': 'string',
    }


def main():
    _datasets_path = '../datasets/liver-disorders'
    _folds_path = '../datasets/folds'

    n_folds = 5

    for _dataset_path in os.listdir(_datasets_path):
        name = _dataset_path.split('.')[0]
        print 'Generating folds for dataset %s' % name
        generate_folds(dataset_path=os.path.join(_datasets_path, _dataset_path), output_folder=_folds_path, n_folds=n_folds)


if __name__ == '__main__':
    main()


