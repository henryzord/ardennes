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
    
    # runs top-down inference algorithm
    if kwargs['tree_depth'] is None:
        td_depth = j48(train_set)
    else:
        td_depth = kwargs['tree_depth']
    target_add = 1. / td_depth
    
    for j in xrange(kwargs['n_run']):  # run the evolutionary process several times
        inst = Ardennes()
        fittest = inst.fit_predict(
            sets=sets,
            n_individuals=kwargs['n_individuals'],
            target_add=target_add,
            inf_thres=0.9,
            diff=kwargs['diff'],
            verbose=kwargs['verbose']
        )
        
        test_acc = fittest.__validate__(sets['test'])
        print 'fold: %d run: %d accuracy: %0.2f' % (fold, j, test_acc)
        
        fold_acc += test_acc
    
    print '%0.d-th fold\tEDA mean accuracy: %0.2f' % (fold, fold_acc / float(kwargs['n_run']))


def j48(train_set):
    """
    Uses J48 algorithm from Weka to get the maximum height of a 100% accuracy decision tree.
    
    :type train_set: pandas.DataFrame
    :param train_set:
    :return:
    """

    import math
    import StringIO
    import weka.core.jvm as jvm
    from weka.classifiers import Classifier
    from weka.core.converters import Loader

    filepath = 'dataset_temp.csv'
    
    train_set.to_csv(filepath, sep=',', quotechar='\"', encoding='utf-8', index=False)
    
    try:
        jvm.start()
    
        loader = Loader(classname='weka.core.converters.CSVLoader')
        data = loader.load_file(filepath)
        data.class_is_last()  # set class as the last attribute
        
        # J-48 parameters:
        # -B : binary splits
        # -M 1: min leaf objects
        # -U : unprunned
        cls = Classifier(classname="weka.classifiers.trees.J48", options=['-U', '-B', '-M', '1'])
        cls.build_classifier(data)
        graph = cls.graph.encode('ascii')

        out = StringIO.StringIO(graph)
        G = nx.Graph(nx.nx_pydot.read_dot(out))
        n_nodes = G.number_of_nodes()
        height = math.ceil(np.log2(n_nodes + 1))

        return height
    except RuntimeError as e:
        jvm.stop()
        os.remove(filepath)
        raise e
    

def main():
    import warnings
    import random

    tree_depth = 7
    random_state = 2
    n_folds = 10
    dataset_path = '/home/henryzord/Projects/forrestTemp/datasets/iris.csv'

    kwargs = {
        'random_state': None,
        'n_individuals': 25,
        'n_run': 1,
        'diff': 0.01,
        'n_folds': n_folds,
        'tree_depth': tree_depth,
        'verbose': True
    }

    if random_state is not None:
        warnings.warn('WARNING: deterministic approach!')
        
        random.seed(random_state)
        np.random.seed(random_state)
    
    if tree_depth is not None:
        warnings.warn('WARNING: hard-coded size of tree!')
    
    df, folds = get_dataset(dataset_path, n_folds=n_folds)  # csv-stored datasets

    for i, (arg_train, arg_test) in enumerate(folds):
        run_fold(i, df, arg_train, arg_test, **kwargs)
        exit(-1)
        
        # TODO for parallel processing
        # t = threading.Thread(target=run_fold, args=(i, df, arg_train, arg_test, kwargs))
        # t.daemon = True
        # t.start()
    
# plt.show()


if __name__ == '__main__':
    main()
