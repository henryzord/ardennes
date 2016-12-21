# coding=utf-8
import copy
import json
import os

import arff
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

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
    import warnings
    warnings.filterwarnings('error')

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    _folds = skf.split(df[df.columns[:-1]], df[df.columns[-1]])

    d_folds = dict()

    for i, (arg_rest, arg_test) in enumerate(_folds):

        x_train, x_val, y_train, y_val = train_test_split(
            df.loc[arg_rest, df.columns[:-1]],
            df.loc[arg_rest, df.columns[-1]],
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
        val_s = df.loc[f[i]['val']]
        test_s = df.loc[f[i]['test']]

        yield train_s, test_s, val_s


def get_dataset_sets(dataset_name, datasets_path, fold_file, output_path='.', merge_val=True):
    """
    Given a dataset (in .arff format), returns its sets (train, val and test).

    :param dataset_name: The name of the dataset (i.e, 'iris' -- do not pass it with a file extension, as in 'iris.arff'
    :param datasets_path: The path in which the dataset is. It is assumed that is a .arff file.
    :param fold_file: A dictionary with the following structure:
        { \n
            \t'0':  # index of the current fold \n
                \t\t'train': [0, 1, 2, ...]  # index of the training instances \n
                \t\t'test': [500, 501, 502, ...]  # index of the test instances \n
                \t\t'val': [600, 601, 602 ...]  # index of the validation instances \n
            \t'1':  \n
                \t\t... \n
        } \n
    :param output_path: optional - File to write csv, csv dataset (one per fold). Defaults to workpath.
    :param merge_val: optional - Whether to merge training and validation sets. In this case, will not output a validation
        csv dataset.
    :return: A dictionary with the same structure as fold_file, except that it contains (relative) file paths to csv
        datasets.
    """
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


def __generate_intermediary_datasets__(datasets_path, folds_path, output_path):
    import weka.core.jvm as jvm
    from weka.core.converters import Loader
    from weka.core.converters import Saver

    jvm.start()

    for dataset in os.listdir(datasets_path):
        dataset_name = dataset.split('.')[0]
        fold_file = json.load(open(os.path.join(folds_path, dataset_name + '.json'), 'r'))

        print 'generating intermediary sets for %s' % dataset_name

        csv_loader = Loader(classname="weka.core.converters.CSVLoader")
        arff_saver = Saver(classname='weka.core.converters.ArffSaver')

        macros = get_dataset_sets(
            dataset_name=dataset_name,
            datasets_path=datasets_path,
            fold_file=fold_file,
            output_path=output_path,
            merge_val=False
        )

        for n_fold, folds_sets in macros.iteritems():
            train_s = csv_loader.load_file(macros[n_fold]['train'])
            test_s = csv_loader.load_file(macros[n_fold]['test'])
            val_s = csv_loader.load_file(macros[n_fold]['val'])

            train_s.relationname = dataset_name
            test_s.relationname = dataset_name
            val_s.relationname = dataset_name

            train_s.class_is_last()
            test_s.class_is_last()
            val_s.class_is_last()

            cpy_train = copy.deepcopy(macros[n_fold]['train']).replace('.csv', '.arff')
            cpy_test = copy.deepcopy(macros[n_fold]['test']).replace('.csv', '.arff')
            cpy_val = copy.deepcopy(macros[n_fold]['val']).replace('.csv', '.arff')

            arff_saver.save_file(train_s, cpy_train)
            arff_saver.save_file(test_s, cpy_test)
            arff_saver.save_file(val_s, cpy_val)

    jvm.stop()


def main():
    _datasets_path = '../datasets/__big__'
    _folds_path = '../datasets/folds'

    n_folds = 10

    for dataset_format in os.listdir(_datasets_path):
        name = dataset_format.split('.')[0]
        df = read_dataset(os.path.join(_datasets_path, dataset_format))

        print 'Genering folds for dataset %s' % name
        generate_folds(df, dataset_name=name, output_folder=_folds_path, n_folds=n_folds)

    __generate_intermediary_datasets__(_datasets_path, _folds_path, output_path='../intermediary')

if __name__ == '__main__':
    main()
