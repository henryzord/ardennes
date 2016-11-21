# coding=utf-8
import json
import random
import warnings
from multiprocessing import Process

import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing.dataset import generate_folds, read_dataset, get_batch
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


def run_fold(fold, dataset, arg_train, arg_test, **kwargs):
    test_s = dataset.iloc[arg_test]  # test set contains both x_test and y_test

    x_train, x_val, y_train, y_val = train_test_split(
        dataset.iloc[arg_train][dataset.columns[:-1]],
        dataset.iloc[arg_train][dataset.columns[-1]],
        test_size=1. / (kwargs['n_folds'] - 1.),
        random_state=kwargs['random_state']
    )

    train_s = x_train
    val_s = x_val
    train_s['class'] = y_train
    val_s['class'] = y_val

    tree_height = __get_tree_height__(train_s, **kwargs)

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
            train=train_s,
            val=val_s,
            test=test_s,
            verbose=kwargs['verbose'],
            output_file=kwargs['output_file'] if kwargs['save_metadata'] else None,
            fold=fold,
            run=j
        )

        _test_acc = inst.validate(test_s, ensemble=kwargs['ensemble'])
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

    mean = accs.mean()
    std = accs.std()

    print '%02.d-th fold\tEDA accuracy: mean %0.2f +- %0.2f' % (fold, mean, std)


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


def do_train(_json_file, mode='cross-validation'):
    print 'training ardennes for %s' % _json_file['dataset_path']

    if mode not in ['cross-validation', 'holdout']:
        raise ValueError('Mode must be either \'cross-validation\' or \'holdout\'!')

    dataset = read_dataset(_json_file['dataset_path'])
    random_state = _json_file['random_state']

    if random_state is not None:
        warnings.warn('WARNING: Using seed=%d (i.e, non-randomic approach)' % random_state)

        random.seed(random_state)
        np.random.seed(random_state)

    if mode == 'cross-validation':
        folds = generate_folds(dataset, n_folds=_json_file['n_folds'], random_state=_json_file['random_state'])

        processes = []
        for i, (arg_train, arg_test) in enumerate(folds):
            p = Process(
                target=run_fold,
                kwargs=dict(fold=i, dataset=dataset, arg_train=arg_train, arg_test=arg_test, **_json_file)
            )
            p.start()
            processes.append(p)

        for process in processes:
            process.join()
    else:
        train_s, val_s, test_s = get_batch(
            dataset, train_size=_json_file['train_size'], random_state=_json_file['random_state']
        )

        run_batch(train_s=train_s, val_s=val_s, test=test_s, **_json_file)


if __name__ == '__main__':
    raise NotImplementedError('must get folds from files!')

    config_file = json.load(open('config.json', 'r'))
    do_train(config_file, mode='holdout')
