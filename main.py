# coding=utf-8
import json
import random
import shutil
import warnings

from datetime import datetime as dt

import numpy as np
import os

from sklearn.tree import DecisionTreeClassifier

from preprocessing.dataset import read_dataset, get_batch, get_fold_iter
from treelib import Ardennes
from treelib import get_max_height
import pandas as pd

__author__ = 'Henry Cagnini'


def __get_tree_height__(_train, **kwargs):
    random_state = kwargs['random_state'] if 'random_state' in kwargs else None
    max_height = get_max_height(_train, random_state)

    if 'tree_height' not in kwargs or kwargs['tree_height'] is None:
        tree_height = max_height
    else:
        c_tree_height = kwargs['tree_height']

        if isinstance(c_tree_height, int):
            tree_height = c_tree_height
        elif isinstance(c_tree_height, str) or isinstance(c_tree_height, unicode):
            tree_height = eval(c_tree_height % max_height)
        else:
            raise TypeError('\'tree_height\' must be either a string, None or an int!')

    return tree_height


def get_baseline_algorithms(names):
    # valid = ['DecisionTreeClassifier']

    algorithms = dict()
    for name in names:
        if name == 'DecisionTreeClassifier':
            algorithms[name] = DecisionTreeClassifier(criterion='entropy')

    return algorithms


def run_fold(n_fold, n_run, train_s, val_s, test_s, config_file):
    tree_height = __get_tree_height__(train_s, **config_file)

    t1 = dt.now()

    _test_acc = -1.
    with Ardennes(
        n_individuals=config_file['n_individuals'],
        decile=config_file['decile'],
        uncertainty=config_file['uncertainty'],
        max_height=tree_height,
        distribution=config_file['distribution'],
        n_iterations=config_file['n_iterations']
    ) as inst:
        inst.fit(
            train=train_s,
            val=val_s,
            test=test_s,
            verbose=config_file['verbose'],
            dataset_name=config_file['dataset_name'],
            output_path=config_file['output_path'] if config_file['save_metadata'] else None,
            fold=n_fold,
            run=n_run
        )

        _test_acc = inst.validate(test_s, ensemble=config_file['ensemble'])

        t2 = dt.now()

    print 'Run %d of fold %d: Test acc: %02.2f, time: %02.2f secs' % (
        n_run, n_fold, _test_acc, (t2 - t1).total_seconds()
    )

    return _test_acc


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
        output_path=kwargs['output_file'] if kwargs['save_metadata'] else None
    )

    test_acc = inst.validate(test, ensemble=kwargs['ensemble'])
    print 'Test accuracy: %0.2f' % test_acc


def do_train(config_file, n_run, evaluation_mode='cross-validation'):
    assert evaluation_mode in ['cross-validation', 'holdout'], \
        ValueError('evaluation_mode must be either \'cross-validation\' or \'holdout!\'')

    dataset_name = config_file['dataset_path'].split('/')[-1].split('.')[0]
    config_file['dataset_name'] = dataset_name
    print 'training ardennes for %s' % dataset_name

    df = read_dataset(config_file['dataset_path'])
    random_state = config_file['random_state']

    if random_state is not None:
        warnings.warn('WARNING: Using non-randomic sampling with seed=%d' % random_state)

        random.seed(random_state)
        np.random.seed(random_state)

    if evaluation_mode == 'cross-validation':
        assert 'folds_path' in config_file, ValueError('Performing a cross-validation is only possible with a json '
                                                       'file for folds! Provide it through the \'folds_path\' '
                                                       'parameter in the configuration file!')

        result_dict = {'folds': dict()}

        folds = get_fold_iter(df, os.path.join(config_file['folds_path'], dataset_name + '.json'))

        for i, (train_s, val_s, test_s) in enumerate(folds):
            print 'Running fold %d for dataset %s' % (i, dataset_name)
            result_dict['folds'][str(i)] = run_fold(
                n_fold=i, n_run=n_run, train_s=train_s, val_s=val_s,
                test_s=test_s, config_file=config_file
            )

        return result_dict

    else:
        raise NotImplementedError('not implemented yet!')

        train_s, val_s, test_s = get_batch(
            df, train_size=config_file['train_size'], random_state=config_file['random_state']
        )

        run_batch(train_s=train_s, val_s=val_s, test=test_s, **config_file)


def crunch_data(results_file):

    n_folds = len(results_file.keys())
    algorithms = results_file[str(0)].keys()
    n_algorithms = len(algorithms)
    # n_runs = len(results_file[str(0)][algorithms[0]])

    df = pd.DataFrame(columns=['algorithm', 'fold', 'acc mean', 'acc std'], index=np.arange(n_algorithms * n_folds))

    count_row = 0
    for n_fold, fold in results_file.iteritems():
        for alg, vec in fold.iteritems():
            acc_mean = np.mean(vec)
            acc_std = np.std(vec)

            df.loc[count_row] = [alg, n_fold, acc_mean, acc_std]
            count_row += 1

    print df

if __name__ == '__main__':
    # _config_file = json.load(open('config.json', 'r'))
    # do_train(_config_file, output_path='metadata', evaluation_mode='cross-validation', n_runs=1)

    _results_file = json.load(open('/home/henry/Desktop/[6 runs] balance-scale.json', 'r'))
    crunch_data(_results_file)
