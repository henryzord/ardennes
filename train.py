# coding=utf-8

from treelib import Ardennes

import json
import warnings
import random
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from treelib.classes import value_check
from treelib import get_max_height

__author__ = 'Henry Cagnini'


def get_folds(df, n_folds=10, random_state=None):
    from sklearn.cross_validation import StratifiedKFold

    Y = df[df.columns[-1]]
    
    folds = StratifiedKFold(Y, n_folds=n_folds, shuffle=True, random_state=random_state)
    return folds


def run_fold(fold, dataset, arg_train, arg_test, **kwargs):
    fold_acc = 0.
    
    test_set = dataset.iloc[arg_test]  # test set contains both x_test and y_test

    cur_date = datetime.now()
    str_date = '%02.d:%02.d %02.d-%02.d-%04.d' % (
        cur_date.hour, cur_date.minute, cur_date.day, cur_date.month, cur_date.year
    )

    x_train, x_val, y_train, y_val = train_test_split(
        dataset.iloc[arg_train][dataset.columns[:-1]],
        dataset.iloc[arg_train][dataset.columns[-1]],
        test_size=1. / (kwargs['n_folds'] - 1.),
        random_state=kwargs['random_state']
    )

    for j in xrange(kwargs['n_runs']):  # run the evolutionary process several times
        file_name = 'fold=%02.d run=%02.d' % (fold, j) + ' ' + str_date + '.csv'

        inst = Ardennes(
            n_individuals=kwargs['n_individuals'],
            threshold=kwargs['decile'],
            uncertainty=kwargs['uncertainty'],
            max_height=kwargs['initial_tree_size'],
            distribution=kwargs['distribution'],
            class_probability=kwargs['class_probability']
        )

        raise NotImplementedError('not implemented yet!')

        inst.fit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            verbose=kwargs['verbose'],
            output_file=file_name if kwargs['save_metadata'] else None
        )
        
        test_acc = inst.validate(test_set, ensemble=kwargs['ensemble'])
        print 'fold: %02.d run: %02.d accuracy: %0.2f' % (fold, j, test_acc)
        
        fold_acc += test_acc
    
    print '%02.d-th fold\tEDA mean accuracy: %0.2f' % (fold, fold_acc / float(kwargs['n_runs']))


def get_batch(dataset, train_size=0.8, random_state=None):
    train, rest = train_test_split(
        dataset,
        train_size=train_size,
        random_state=random_state
    )

    val, test = train_test_split(
        rest,
        test_size=0.5,
        random_state=random_state
    )

    return train, val, test


def main(json_file, mode='batch'):
    value_check(mode, ['batch', 'folds'])

    with open(json_file, 'r') as f:
        kwargs = json.load(f)

    if kwargs['random_state'] is not None:
        warnings.warn('WARNING: deterministic approach!')
        
        random.seed(kwargs['random_state'])
        np.random.seed(kwargs['random_state'])

    dataset = pd.read_csv(kwargs['dataset_path'], sep=',')

    if mode == 'folds':
        folds = get_folds(dataset, n_folds=kwargs['n_folds'], random_state=kwargs['random_state'])

        for i, (arg_train, arg_test) in enumerate(folds):
            run_fold(fold=i, dataset=dataset, arg_train=arg_train, arg_test=arg_test, **kwargs)
    else:
        train, val, test = get_batch(
            dataset, train_size=kwargs['train_size'], random_state=kwargs['random_state']
        )

        if 'tree_height' not in kwargs:
            try:
                tree_height = get_max_height(train, kwargs['random_state'])
            except ValueError as ve:
                tree_height = kwargs['tree_height']
        else:
            tree_height = kwargs['tree_height']

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
            verbose=kwargs['verbose'],
            output_file=kwargs['output_file'] if kwargs['save_metadata'] else None,
            metadata_path=kwargs['metadata_path']
        )

        test_acc = inst.validate(test, ensemble=kwargs['ensemble'])
        print 'Test accuracy: %0.2f' % test_acc

        inst.plot(metadata_path=kwargs['metadata_path'])

if __name__ == '__main__':
    _json_file = 'input.json'
    main(_json_file, mode='batch')
