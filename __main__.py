# coding=utf-8
import json
import random
import warnings

from datetime import datetime as dt

import numpy as np
import os

from sklearn.tree import DecisionTreeClassifier

from preprocessing.dataset import read_dataset, get_batch, get_fold_iter
from treelib import Ardennes
from treelib import get_max_height

__author__ = 'Henry Cagnini'


def __get_tree_height__(_train, **kwargs):
    if 'tree_height' not in kwargs or kwargs['tree_height'] is None:
        try:
            tree_height = get_max_height(_train, kwargs['random_state'])
        except ValueError as ve:
            tree_height = kwargs['tree_height']
    else:
        tree_height = kwargs['tree_height']

    return tree_height


def get_baseline_algorithms(names):
    # valid = ['DecisionTreeClassifier']

    algorithms = dict()
    for name in names:
        if name == 'DecisionTreeClassifier':
            algorithms[name] = DecisionTreeClassifier(criterion='entropy')

    return algorithms


def run_fold(n_fold, train_s, val_s, test_s, n_runs, config_file):
    def run_ardennes(run, l_algorithms, **l_kwargs):
        for name, clf in l_algorithms.iteritems():
            clf.fit(
                train_s[train_s.columns[:-1]], train_s[train_s.columns[-1]]
            )  # type: DecisionTreeClassifier
            acc = clf.score(test_s[test_s.columns[:-1]], test_s[test_s.columns[-1]])
            alg_results[name][run] = acc

        t1 = dt.now()

        inst = Ardennes(
            n_individuals=l_kwargs['n_individuals'],
            decile=l_kwargs['decile'],
            uncertainty=l_kwargs['uncertainty'],
            max_height=tree_height,
            distribution=l_kwargs['distribution'],
            n_iterations=l_kwargs['n_iterations']
        )

        inst.fit(
            train=train_s,
            val=val_s,
            test=test_s,
            verbose=l_kwargs['verbose'],
            output_file=l_kwargs['output_file'] if l_kwargs['save_metadata'] else None,
            fold=n_fold,
            run=j
        )

        _test_acc = inst.validate(test_s, ensemble=l_kwargs['ensemble'])
        t2 = dt.now()
        ardennes_accs[run] = _test_acc
        ardennes_times[run] = (t2 - t1).total_seconds()

        print 'Run %d of fold %d: Test acc: %02.2f, time: %02.2f secs' % (
            run, n_fold, _test_acc, ardennes_times[run]
        )

    tree_height = __get_tree_height__(train_s, **config_file)

    base_alg_names = config_file['baseline_algorithms']
    algorithms = get_baseline_algorithms(base_alg_names)

    ardennes_accs = np.empty(n_runs, dtype=np.float32).tolist()
    ardennes_times = np.empty(n_runs, dtype=np.float32).tolist()

    alg_results = {name: np.empty(n_runs, dtype=np.float32).tolist() for name in algorithms.iterkeys()}

    for j in xrange(n_runs):
        run_ardennes(j, algorithms, **config_file)

    acc_mean = np.mean(ardennes_accs)
    acc_std = np.std(ardennes_accs)

    time_mean = np.mean(ardennes_times)
    time_std = np.std(ardennes_times)

    print 'Fold %d (%d runs): accuracy %02.2f +- %02.2f\ttime: %02.2f +- %02.2f secs' % (
        n_fold, n_runs, acc_mean, acc_std, time_mean, time_std
    )

    alg_results['ardennes'] = ardennes_accs

    return alg_results


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


def do_train(config_file, output_folder=None, mode='cross-validation'):
    assert mode in ['cross-validation', 'holdout'], \
        ValueError('Mode must be either \'cross-validation\' or \'holdout!\'')

    dataset_name = config_file['dataset_path'].split('/')[-1].split('.')[0]
    print 'training ardennes for %s' % dataset_name

    df = read_dataset(config_file['dataset_path'])
    random_state = config_file['random_state']

    if random_state is not None:
        warnings.warn('WARNING: Using non-randomic sampling with seed=%d' % random_state)

        random.seed(random_state)
        np.random.seed(random_state)

    if mode == 'cross-validation':
        assert 'folds_path' in config_file, ValueError('Performing a cross-validation is only possible with a json '
                                                       'file for folds! Provide it through the \'folds_path\' '
                                                       'parameter in the configuration file!')

        folds = get_fold_iter(df, os.path.join(config_file['folds_path'], dataset_name + '.json'))

        all_fold_results = dict()

        for i, (train_s, val_s, test_s) in enumerate(folds):
            print 'Running fold %d for dataset %s' % (i, dataset_name)
            all_fold_results[i] = run_fold(
                n_fold=i, train_s=train_s, val_s=val_s,
                test_s=test_s, n_runs=config_file['n_runs'], config_file=config_file
            )

        if output_folder is not None:
            json.dump(all_fold_results, open(os.path.join(output_folder, '%s.json' % dataset_name), 'w'), indent=2)

    else:
        train_s, val_s, test_s = get_batch(
            df, train_size=config_file['train_size'], random_state=config_file['random_state']
        )

        run_batch(train_s=train_s, val_s=val_s, test=test_s, **config_file)


def crunch_data(results_file):
    for fold in sorted(results_file.keys()):
        for alg, vec in results_file[fold].iteritems():
            print '%s fold %d: %02.2f +- %02.2f' % (alg, int(fold), np.mean(vec), np.std(vec))

if __name__ == '__main__':
    _config_file = json.load(open('config.json', 'r'))
    do_train(_config_file, output_folder='metadata', mode='cross-validation')

    # _results_file = json.load(open('metadata/results.json', 'r'))
    # crunch_data(_results_file)
