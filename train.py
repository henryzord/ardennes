# coding=utf-8

import os
import pandas as pd
from sklearn.cross_validation import train_test_split
from evolution import Ardennes
import numpy as np
import networkx as nx

__author__ = 'Henry Cagnini'


def get_folds(df, n_folds=10, random_state=None):
    from sklearn.cross_validation import StratifiedKFold
    
    Y = df[df.columns[-1]]
    
    folds = StratifiedKFold(Y, n_folds=n_folds, shuffle=True, random_state=random_state)
    return folds


def get_dataset(path, n_folds=10, random_state=None):
    df = pd.read_csv(path, sep=',')
    folds = get_folds(df, n_folds=n_folds, random_state=random_state)
    return df, folds


def run_fold(fold, df, arg_train, arg_test, **kwargs):
    fold_acc = 0.
    
    # x_test = df.iloc[arg_test][df.columns[:-1]]
    # y_test = df.iloc[arg_test][df.columns[-1]]
    
    test_set = df.iloc[arg_test]  # test set contains both x_test and y_test
    
    x_train, x_val, y_train, y_val = train_test_split(
        df.iloc[arg_train][df.columns[:-1]],
        df.iloc[arg_train][df.columns[-1]],
        test_size=1. / (kwargs['n_folds'] - 1.),
        random_state=kwargs['random_state']
    )
    
    train_set = x_train.join(y_train)  # type: pd.DataFrame
    val_set = x_val.join(y_val)  # type: pd.DataFrame
    
    sets = {'train': train_set, 'val': val_set, 'test': test_set}
    
    for j in xrange(kwargs['n_runs']):  # run the evolutionary process several times
        inst = Ardennes(
            n_individuals=kwargs['n_individuals'],
            threshold=kwargs['decile'],
            uncertainty=kwargs['uncertainty']
        )
        
        fittest = inst.fit_predict(
            sets=sets,
            verbose=kwargs['verbose']
        )
        
        test_acc = fittest.__validate__(sets['test'])
        print 'fold: %02.d run: %02.d accuracy: %0.2f' % (fold, j, test_acc)
        
        fold_acc += test_acc
    
    print '%02.d-th fold\tEDA mean accuracy: %0.2f' % (fold, fold_acc / float(kwargs['n_runs']))


def main():
    import warnings
    import random
    
    random_state = 1
    n_folds = 10
    n_runs = 1
    dataset_path = 'datasets/iris.csv'
    
    kwargs = {
        'n_individuals': 50,
        'decile': 0.9,
        'uncertainty': 0.01,
        'verbose': True,
        'n_folds': n_folds,
        'n_runs': n_runs,
        'random_state': random_state
    }
    
    if random_state is not None:
        warnings.warn('WARNING: deterministic approach!')
        
        random.seed(random_state)
        np.random.seed(random_state)
    
    df, folds = get_dataset(dataset_path, n_folds=n_folds)  # csv-stored datasets
    
    for i, (arg_train, arg_test) in enumerate(folds):
        run_fold(fold=i, df=df, arg_train=arg_train, arg_test=arg_test, **kwargs)
        warnings.warn('WARNING: exiting after first fold!')
        exit(0)

if __name__ == '__main__':
    main()
