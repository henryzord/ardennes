# coding=utf-8
from multiprocessing import Process, Array

from treelib import Ardennes

import json
import warnings
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from treelib import get_max_height
import arff

__author__ = 'Henry Cagnini'


def __comon__(json_file):
    if json_file['random_state'] is not None:
        warnings.warn('WARNING: Using seed=%d (i.e, non-randomic approach)' % json_file['random_state'])

        random.seed(json_file['random_state'])
        np.random.seed(json_file['random_state'])

    dataset_path = json_file['dataset_path']
    dataset_type = dataset_path.split('.')[-1].strip()

    if dataset_type == 'csv':
        dataset = pd.read_csv(json_file['dataset_path'], sep=',')
    elif dataset_type == 'arff':
        af = arff.load(open(dataset_path, 'r'))
        dataset = pd.DataFrame(af['data'], columns=[x[0] for x in af['attributes']])
    else:
        raise TypeError('Invalid type for dataset! Must be either \'csv\' or \'arff\'!')

    return dataset


def __get_tree_height__(train, **kwargs):
    if 'tree_height' not in kwargs or kwargs['tree_height'] is None:
        try:
            tree_height = get_max_height(train, kwargs['random_state'])
        except ValueError as ve:
            tree_height = kwargs['tree_height']
    else:
        tree_height = kwargs['tree_height']

    return tree_height


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


def get_folds(df, n_folds=10, random_state=None):
    from sklearn.cross_validation import StratifiedKFold

    _Y = df[df.columns[-1]]
    
    _folds = StratifiedKFold(_Y, n_folds=n_folds, shuffle=True, random_state=random_state)
    return _folds


def run_fold(fold, dataset, arg_train, arg_test, **kwargs):
    fold_acc = 0.

    test_set = dataset.iloc[arg_test]  # test set contains both x_test and y_test

    x_train, x_val, y_train, y_val = train_test_split(
        dataset.iloc[arg_train][dataset.columns[:-1]],
        dataset.iloc[arg_train][dataset.columns[-1]],
        test_size=1. / (kwargs['n_folds'] - 1.),
        random_state=kwargs['random_state']
    )

    train = x_train
    val = x_val
    train['class'] = y_train
    val['class'] = y_val

    tree_height = __get_tree_height__(train, **kwargs)

    # accs = Array('f', range(kwargs['n_runs']))
    accs = np.empty(kwargs['n_runs'], dtype=np.float32)

    def run_ardennes(run, arr, **kwargs):
        inst = Ardennes(
            n_individuals=kwargs['n_individuals'],
            decile=kwargs['decile'],
            uncertainty=kwargs['uncertainty'],
            max_height=tree_height,
            distribution=kwargs['distribution'],
            n_iterations=kwargs['n_iterations']
        )

        inst.fit(
            train=train,
            val=val,
            test=test_set,
            verbose=kwargs['verbose'],
            output_file=kwargs['output_file'] if kwargs['save_metadata'] else None,
            fold=fold,
            run=j
        )

        _test_acc = inst.validate(test_set, ensemble=kwargs['ensemble'])
        arr[run] = _test_acc

    # processes = []
    for j in xrange(kwargs['n_runs']):
        run_ardennes(j, accs, **kwargs)

    #     p = Process(target=run_ardennes, args=(j, accs), kwargs=kwargs)
    #     p.start()
    #     processes.append(p)
    #
    # for process in processes:
    #     process.join()

    fold_acc = sum(accs)

    print '%02.d-th fold\tEDA mean accuracy: %0.2f' % (fold, fold_acc / float(kwargs['n_runs']))


def run_batch(train_s, val_s, test, **kwargs):
    tree_height = __get_tree_height__(train_s, **kwargs)

    inst = Ardennes(
        n_individuals=kwargs['n_individuals'],
        decile=kwargs['decile'],
        uncertainty=kwargs['uncertainty'],
        max_height=tree_height,
        distribution=kwargs['distribution'],
        n_iterations=kwargs['n_iterations']
    )

    inst.fit(
        train=train_s,
        val=val_s,
        test=test,
        verbose=kwargs['verbose'],
        output_file=kwargs['output_file'] if kwargs['save_metadata'] else None
    )

    test_acc = inst.validate(test, ensemble=kwargs['ensemble'])
    print 'Test accuracy: %0.2f' % test_acc


def train(json_file, mode='cross-validation'):
    if mode not in ['cross-validation', 'holdout']:
        raise ValueError('Mode must be either \'cross-validation\' or \'holdout\'!')

    dataset = __comon__(json_file)

    if mode == 'cross-validation':
        folds = get_folds(dataset, n_folds=json_file['n_folds'], random_state=json_file['random_state'])

        processes = []
        for i, (arg_train, arg_test) in enumerate(folds):
            p = Process(
                target=run_fold,
                kwargs=dict(fold=i, dataset=dataset, arg_train=arg_train, arg_test=arg_test, **json_file)
            )
            p.start()
            processes.append(p)

        for process in processes:
            process.join()
    else:
        train_s, val_s, test_s = get_batch(
            dataset, train_size=json_file['train_size'], random_state=json_file['random_state']
        )

        run_batch(train_s=train_s, val_s=val_s, test=test_s, **json_file)


if __name__ == '__main__':
    json_file = json.load(open('input.json', 'r'))

    train(json_file, mode='cross-validation')
