# coding=utf-8
import json
import os
import random
import warnings
from datetime import datetime as dt
from multiprocessing import Process, Manager
import itertools as it

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from preprocessing.dataset import read_dataset, get_batch, get_fold_iter
from treelib import Ardennes
from treelib import get_max_height

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


def run_fold(n_fold, n_run, train_s, val_s, test_s, config_file, **kwargs):
    try:
        random_state = kwargs['random_state']
    except KeyError:
        random_state = None

    if random_state is not None:
        warnings.warn('WARNING: Using non-randomic sampling with seed=%d' % random_state)

    random.seed(random_state)
    np.random.seed(random_state)

    tree_height = __get_tree_height__(train_s, **config_file)

    t1 = dt.now()

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
            output_path=config_file['output_path'] if 'output_path' in config_file else None,
            fold=n_fold,
            run=n_run
        )

        _test_acc = inst.validate(test_s, ensemble=config_file['ensemble'])
        _tree_height = inst.tree_height

        t2 = dt.now()

    print 'Run %d of fold %d: Test acc: %02.2f, time: %02.2f secs' % (
        n_run, n_fold, _test_acc, (t2 - t1).total_seconds()
    )

    if 'dict_manager' in kwargs:
        kwargs['dict_manager'][n_fold] = dict(acc=_test_acc, height=_tree_height)

    return _test_acc


def do_train(config_file, n_run, evaluation_mode='cross-validation'):
    """

    :param config_file:
    :param n_run:
    :param evaluation_mode:
    :return:
    """

    assert evaluation_mode in ['cross-validation', 'holdout'], \
        ValueError('evaluation_mode must be either \'cross-validation\' or \'holdout!\'')

    dataset_name = config_file['dataset_path'].split('/')[-1].split('.')[0]
    config_file['dataset_name'] = dataset_name
    print 'training ardennes for %s' % dataset_name

    df = read_dataset(config_file['dataset_path'])
    random_state = config_file['random_state']

    if evaluation_mode == 'cross-validation':
        assert 'folds_path' in config_file, ValueError('Performing a cross-validation is only possible with a json '
                                                       'file for folds! Provide it through the \'folds_path\' '
                                                       'parameter in the configuration file!')

        result_dict = {'folds': dict()}

        folds = get_fold_iter(df, os.path.join(config_file['folds_path'], dataset_name + '.json'))

        manager = Manager()
        dict_manager = manager.dict()

        processes = []

        for i, (train_s, val_s, test_s) in enumerate(folds):
            warnings.warn('WARNING: appending train and val sets!')

            train_s = train_s.append(val_s, ignore_index=True)
            val_s = train_s

            p = Process(
                target=run_fold, kwargs=dict(
                    n_fold=i, n_run=n_run, train_s=train_s, val_s=val_s,
                    test_s=test_s, config_file=config_file, dict_manager=dict_manager, random_state=random_state
                )
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        result_dict['folds'] = dict(dict_manager)
        return result_dict

    else:
        train_s, val_s, test_s = get_batch(
            df, train_size=config_file['train_size'], random_state=config_file['random_state']
        )
        run_fold(n_fold=0, n_run=0, train_s=train_s, val_s=val_s, test_s=test_s, config_file=config_file, random_state=random_state)


def crunch_result_file(results_file, output_file=None):

    n_runs = len(results_file['runs'].keys())
    some_run = results_file['runs'].keys()[0]
    some_dataset = results_file['runs'][some_run].keys()[0]
    n_datasets = len(results_file['runs'][some_run].keys())
    n_folds = len(results_file['runs'][some_run][some_dataset]['folds'].keys())

    df = pd.DataFrame(
        columns=['run', 'dataset', 'fold', 'acc', 'height'],
        index=np.arange(n_runs * n_datasets * n_folds)
    )

    count_row = 0
    for n_run, run in results_file['runs'].iteritems():
        for dataset_name, dataset in run.iteritems():
            for n_fold, v in dataset['folds'].iteritems():
                acc = v['acc']
                height = v['height']
                df.loc[count_row] = [int(n_run), str(dataset_name), int(n_fold), float(acc), float(height)]
                count_row += 1

    df['acc'] = df['acc'].astype(np.float)
    df['dataset'] = df['dataset'].astype(np.object)
    df['run'] = df['run'].astype(np.int)
    df['fold'] = df['fold'].astype(np.int)
    df['height'] = df['height'].astype(np.float)

    print df

    grouped = df.groupby(by=['dataset'])['acc', 'height']
    final = grouped.aggregate([np.mean, np.std])

    print final

    if output_file is not None:
        final.to_csv(output_file, sep=',', quotechar='\"')


def crunch_population_data(path_results):
    from matplotlib import pyplot as plt

    def best_tree_height(_dirs):
        def is_csv(_f):
            return _f.split('.')[-1] == 'csv'

        def get_heights(_f):
            def proper_get(df):
                _h = df.loc[(df['iteration'] == df['iteration'].max())]
                _h = _h.loc[_h['test accuracy'] == _h['test accuracy'].max()]['tree height']
                return _h

            fpcsv = os.path.join(fp, _f)

            df = pd.read_csv(fpcsv, delimiter=',')
            if 'iteration' not in df.columns:
                df = pd.read_csv(fpcsv, delimiter=',',
                                 names=['individual', 'iteration', 'validation accuracy', 'tree height',
                                        'test accuracy'])
            l_heights = proper_get(df)
            return l_heights

        heights = []

        for dir in _dirs:
            fp = os.path.join(path_results, dir)
            csv_files = [f for f in os.listdir(fp) if is_csv(f)]
            for csv_f in csv_files:
                heights.extend(get_heights(csv_f))

        plt.figure()

        n, bins, patches = plt.hist(heights, facecolor='green')  # 50, normed=1, alpha=0.75)

        plt.xlabel('Heights')
        plt.ylabel('Quantity')
        plt.title('heights')
        plt.grid(True)

    dirs = [f for f in os.listdir(path_results) if not os.path.isfile(os.path.join(path_results, f))]
    best_tree_height(dirs)
    plt.show()


def optimize_params(config_file, n_tries=10):
    config_file['verbose'] = False

    dataset_name = config_file['dataset_path'].split('/')[-1].split('.')[0]

    params = pd.DataFrame(index=np.arange(n_tries), columns=['n_individuals', 'n_iterations', 'tree_height', 'decile', 'acc mean', 'acc std'])

    for some_try in xrange(n_tries):
        config_file['n_individuals'] = np.random.randint(100, 200 + 1)
        config_file['n_iterations'] = np.random.randint(10, 50 + 1)
        config_file['tree_height'] = np.random.randint(5, 7 + 1)
        config_file['decile'] = np.random.randint(50, 90 + 1) / 100.

        params.iloc[some_try]['n_individuals', 'n_iterations', 'tree_height', 'decile'] = [
            config_file['n_individuals'],
            config_file['n_iterations'],
            config_file['tree_height'],
            config_file['decile']
        ]

        print 'n_individuals: %d n_iterations: %d tree_height: %d decile: %.2f' % \
              tuple(params.iloc[some_try][['n_individuals', 'n_iterations', 'tree_height', 'decile']])

        dict_results = do_train(config_file=config_file, n_run=0, evaluation_mode='cross-validation')
        accs = np.array(dict_results['folds'].values(), dtype=np.float32)
        print 'acc: %02.2f +- %02.2f' % (accs.mean(), accs.std())

        params.iloc[some_try]['acc mean', 'acc std'] = [accs.mean(), accs.std()]

    params.to_csv('parametrization_%s.csv' % dataset_name, index=False)


def crunch_parametrization(path_file):
    import plotly.graph_objs as go
    from plotly.offline import plot

    df = pd.read_csv(path_file)  # type: pd.DataFrame

    attrX = 'n_individuals'
    attrY = 'decile'
    attrZ = 'n_iterations'

    trace2 = go.Scatter3d(
        x=df[attrX],
        y=df[attrY],
        z=df[attrZ],
        mode='markers',
        text=['%s: %2.2f<br>%s: %2.2f<br>%s: %2.2f<br>mean acc: %0.2f' %
              (attrX, info[attrX], attrY, info[attrY], attrZ, info[attrZ], info['acc mean']) for (index, info) in df.iterrows()
              ],
        hoverinfo='text',
        marker=dict(
            color=df['acc mean'],
            colorscale='RdBu',
            colorbar=dict(
                title='Mean accuracy',
            ),
            # cmin=0.,  # minimum color value
            # cmax=1.,  # maximum color value
            # cauto=False,  # do not automatically fit color values
            size=12,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.9
        )
    )
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        scene=dict(
           xaxis=dict(title=attrX),
           yaxis=dict(title=attrY),
           zaxis=dict(title=attrZ)
        )
    )
    fig = go.Figure(data=[trace2], layout=layout)
    plot(fig, filename='.parametrization.html')


if __name__ == '__main__':
    _config_file = json.load(open('config.json', 'r'))

    # --------------------------------------------------- #
    # optimize_params(_config_file, 50)
    # --------------------------------------------------- #
    # crunch_parametrization('parametrization_hayes-roth-full.csv')
    # --------------------------------------------------- #

    global_accs = np.empty(10 * 10, dtype=np.float)

    _evaluation_mode = 'cross-validation'
    _dict_results = do_train(config_file=_config_file, n_run=0, evaluation_mode=_evaluation_mode)

    if _evaluation_mode == 'cross-validation':
        _accs = np.array([x['acc'] for x in _dict_results['folds'].itervalues()], dtype=np.float32)
        _heights = np.array([x['height'] for x in _dict_results['folds'].itervalues()], dtype=np.float32)

        print 'acc: %02.2f +- %02.2f\ttree height: %02.2f +- %02.2f' % (
            _accs.mean(), _accs.std(), _heights.mean(), _heights.std()
        )

    # --------------------------------------------------- #

    # _results_file = json.load(
    #     open('/home/henry/Projects/ardennes/metadata/past_runs/j48/j48_results.json', 'r')
    # )
    # crunch_result_file(_results_file, output_file='j48_results.csv')

    # --------------------------------------------------- #

    # _results_path = '/home/henry/Projects/ardennes/metadata/past_runs/[10 runs 10 folds] ardennes'
    # crunch_ensemble(_results_path)
