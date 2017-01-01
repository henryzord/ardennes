# coding=utf-8
import csv
import json
import os
import random
import shutil
import warnings
from datetime import datetime as dt
from multiprocessing import Process, Manager

import pandas as pd
from sklearn.metrics import *

from preprocessing.dataset import read_dataset, get_batch, get_fold_iter
from treelib import Ardennes
from treelib.node import *

import operator as op

__author__ = 'Henry Cagnini'


def run_fold(n_fold, n_run, full, train_s, val_s, test_s, config_file, **kwargs):
    try:
        random_state = kwargs['random_state']
    except KeyError:
        random_state = None

    if random_state is not None:
        warnings.warn('WARNING: Using non-randomic sampling with seed=%d' % random_state)

    random.seed(random_state)
    np.random.seed(random_state)

    tree_height = config_file['tree_height']

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
            full=full,
            train=train_s,
            val=val_s,
            test=test_s,
            verbose=config_file['verbose'],
            dataset_name=config_file['dataset_name'],
            output_path=config_file['output_path'] if 'output_path' in config_file else None,
            fold=n_fold,
            run=n_run
        )

        y_train_true = train_s[train_s.columns[-1]]
        y_val_true = val_s[val_s.columns[-1]]
        y_test_true = test_s[test_s.columns[-1]]

        y_pred_train = inst.predict(train_s, ensemble=config_file['ensemble'])
        y_pred_val = inst.predict(val_s, ensemble=config_file['ensemble'])
        y_pred_test = inst.predict(test_s, ensemble=config_file['ensemble'])

        _train_acc = accuracy_score(y_train_true, y_pred_train)
        _val_acc = accuracy_score(y_val_true, y_pred_val)
        _test_acc = accuracy_score(y_test_true, y_pred_test)  # accuracy

        _test_prc = precision_score(y_test_true, y_pred_test, average='micro')  # precision
        _test_f1s = f1_score(y_test_true, y_pred_test, average='micro')  # f1 measure

        _tree_height = inst.tree_height

        t2 = dt.now()

    print 'Run %d of fold %d: Test acc: %02.2f, time: %02.2f secs' % (
        n_run, n_fold, _test_acc, (t2 - t1).total_seconds()
    )

    if 'dict_manager' in kwargs:
        kwargs['dict_manager'][n_fold] = dict(
            train_acc=_train_acc,
            val_acc=_val_acc,
            acc=_test_acc,
            f1_score=_test_f1s,
            precision=_test_prc,
            height=_tree_height,
            # y_pred_test=list(y_pred_test),
            # y_pred_train=list(y_pred_train),
            # y_pred_val=list(y_pred_val)
        )

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

    def __append__(_train_s, _val_s, __config_file):
        from sklearn.model_selection import train_test_split
        warnings.warn('WARNING: Using %2.2f of data for training' % __config_file['train_size'])

        mass = _train_s.append(_val_s, ignore_index=False)
        _train_s, _val_s = train_test_split(mass, train_size=__config_file['train_size'])

        return _train_s, _val_s

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
            train_s, val_s = __append__(train_s, val_s, config_file)

            p = Process(
                target=run_fold, kwargs=dict(
                    n_fold=i, n_run=n_run, train_s=train_s, val_s=val_s,
                    test_s=test_s, config_file=config_file, dict_manager=dict_manager, random_state=random_state,
                    full=df
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

        train_s, val_s = __append__(train_s, val_s, config_file)

        run_fold(
            n_fold=0, n_run=0, train_s=train_s, val_s=val_s,
            test_s=test_s, config_file=config_file, random_state=random_state,
            full=df
        )


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


def grid_optimizer(config_file, datasets_path, output_path):
    from evaluate import evaluate_ardennes

    config_file['verbose'] = False

    datasets = os.listdir(datasets_path)

    range_individuals = [200]
    range_tree_height = [7]
    range_iterations = [50, 60, 70, 80, 90, 100]
    range_decile = [.5, .55, .6, .65, .70, .75, .8, .85, .9, .95]
    n_runs = 10

    n_opts = reduce(
        op.mul, map(
            len,
            [range_individuals, range_tree_height, range_iterations, range_decile],
        )
    )

    count_row = 0

    for n_individuals in range_individuals:
        for tree_height in range_tree_height:
            for n_iterations in range_iterations:
                for decile in range_decile:

                    _partial_str = '[n_individuals:%d][n_iterations:%d][tree_height:%d][decile:%d]' % \
                                   (n_individuals, n_iterations, tree_height, int(decile * 100))

                    print 'opts: %02.d/%2.d' % (count_row, n_opts) + ' ' + _partial_str

                    _write_path = os.path.join(output_path, _partial_str)
                    if os.path.exists(_write_path):
                        shutil.rmtree(_write_path)
                    os.mkdir(_write_path)

                    config_file['n_individuals'] = n_individuals
                    config_file['n_iterations'] = n_iterations
                    config_file['tree_height'] = tree_height
                    config_file['decile'] = decile
                    config_file['n_runs'] = n_runs

                    evaluate_ardennes(
                        datasets_path=_datasets_path,
                        config_file=config_file,
                        output_path=_write_path,
                        validation_mode='cross-validation'
                    )

                    count_row += 1

                    print '%02.d/%02.d' % (count_row, n_opts)


# noinspection PyUnresolvedReferences
def crunch_parametrization(path_file):
    import plotly.graph_objs as go
    from plotly.offline import plot

    full = pd.read_csv(path_file)  # type: pd.DataFrame
    df = full

    attrX = 'n_individuals'
    attrY = 'decile'
    attrZ = 'n_iterations'

    print 'attributes (x, y, z): (%s, %s, %s)' % (attrX, attrY, attrZ)

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
    plot(fig, filename='parametrization.html')


def crunch_graphical_model(pgm_path, path_datasets):
    from matplotlib import pyplot as plt
    from networkx.drawing.nx_agraph import graphviz_layout
    from networkx import nx
    import itertools as it
    import plotly.graph_objs as go
    from plotly.offline import plot

    def build_graph(series):
        G = nx.DiGraph()

        node_labels = dict()

        for node_id in xrange(series.shape[1]):
            probs = series[:, node_id]

            G.add_node(
                node_id,
                attr_dict=dict(
                    color=max(probs),
                    probs='<br>'.join(['%2.3f : %s' % (y, x) for x, y in it.izip(columns, probs)])
                )
            )
            parent = get_parent(node_id)
            if parent is not None:
                G.add_edge(parent, node_id)

            node_labels[node_id] = node_id

        return G

    def build_edges(_G):
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=go.Line(width=0.5, color='#999'),
            hoverinfo='none',
            mode='lines',
            name='edges'
        )

        for edge in _G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += [x0, x1, None]
            edge_trace['y'] += [y0, y1, None]

        return edge_trace

    def build_nodes(_G, _generation):
        nodes = _G.nodes(data=True)

        _node_trace = go.Scatter(
            x=[pos[node[0]][0] for node in nodes],
            y=[pos[node[0]][1] for node in nodes],
            name='gen %d' % _generation,
            text=[x[1]['probs'] for x in nodes],
            mode='markers',
            visible=True if _generation == 0 else 'legendonly',
            hoverinfo='text',
            marker=go.Marker(
                showscale=True,
                color=[x[1]['color'] for x in nodes],
                colorscale='RdBu',
                colorbar=dict(
                    title='Assurance',
                    xpad=100,
                ),
                cmin=0.,  # minimum color value
                cmax=1.,  # maximum color value
                cauto=False,  # do not automatically fit color values
                reversescale=False,
                size=15,
                line=dict(
                    width=2
                )
            )
        )
        return _node_trace

    sep = '\\' if os.name == 'nt' else '/'

    dataset_name = pgm_path.split(sep)[-1].split('_')[0]

    dataset = read_dataset(os.path.join(path_datasets, dataset_name + '.arff'))
    columns = dataset.columns
    n_columns = dataset.shape[1]
    del dataset

    data = []

    with open(pgm_path, 'r') as f:
        csv_w = csv.reader(f, delimiter=',', quotechar='\"')
        for generation, line in enumerate(csv_w):
            series = np.array(line, dtype=np.float).reshape(n_columns, -1)

            G = build_graph(series)

            pos = graphviz_layout(G, root=0, prog='dot')

            if generation == 0:
                data.append(build_edges(G))

            node_trace = build_nodes(G, generation)
            data += [node_trace]

        fig = go.Figure(
            data=go.Data(data),
            layout=go.Layout(
                title='Probabilistic Graphical Model<br>Dataset %s' % dataset_name,
                titlefont=dict(size=16),
                showlegend=True,
                hovermode='closest',

                xaxis=go.XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=go.YAxis(showgrid=False, zeroline=False, showticklabels=False),
            )
        )

        plot(fig, filename=pgm_path.split(sep)[-1] + '.html')
        # plot(data, filename=pgm_path.split(sep)[-1] + '.html')

if __name__ == '__main__':
    _config_file = json.load(open('config.json', 'r'))
    _datasets_path = 'datasets/numerical'

    # --------------------------------------------------- #
    # crunch_graphical_model(
    #     '/home/henryzord/Projects/ardennes/metadata/liver-disorders/liver-disorders_pgm_fold_000_run_000.csv',
    #     _datasets_path
    # )
    # --------------------------------------------------- #
    # grid_optimizer(_config_file, _datasets_path, output_path='/home/henry/Desktop/parametrizations')
    # --------------------------------------------------- #
    # crunch_parametrization('parametrization_hayes-roth-full.csv')
    # --------------------------------------------------- #
    # _results_file = json.load(
    #     open('/home/henry/Desktop/del/results.json', 'r')
    # )
    # crunch_result_file(_results_file, output_file='results.csv')
    # --------------------------------------------------- #
    # _results_path = '/home/henry/Projects/ardennes/metadata/past_runs/[10 runs 10 folds] ardennes'
    # crunch_ensemble(_results_path)
    # --------------------------------------------------- #
    _evaluation_mode = 'holdout'

    _dict_results = do_train(config_file=_config_file, n_run=0, evaluation_mode=_evaluation_mode)

    if _evaluation_mode == 'cross-validation':
        _accs = np.array([x['acc'] for x in _dict_results['folds'].itervalues()], dtype=np.float32)
        _heights = np.array([x['height'] for x in _dict_results['folds'].itervalues()], dtype=np.float32)

        print 'acc: %02.2f +- %02.2f\ttree height: %02.2f +- %02.2f' % (
            _accs.mean(), _accs.std(), _heights.mean(), _heights.std()
        )
    # --------------------------------------------------- #
