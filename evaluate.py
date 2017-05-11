# coding=utf-8

"""
Performs tests in an 'industrial' fashion.
"""

import StringIO
import csv
import itertools as it
import json
import os
import shutil
import time
import warnings
from datetime import datetime as dt
from multiprocessing import Process, Manager

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from preprocessing.dataset import load_dataframe, load_arff
from treelib import Ardennes
from treelib.node import *
from treelib.utils import DatabaseHandler

__author__ = 'Henry Cagnini'


# noinspection PyUnresolvedReferences
def evaluate_j48(datasets_path, intermediary_path):
    # for examples on how to use this function, refer to
    # http://pythonhosted.org/python-weka-wrapper/examples.html#build-classifier-on-dataset-output-predictions
    import weka.core.jvm as jvm
    from weka.core.converters import Loader
    from weka.classifiers import Classifier
    from sklearn.metrics import precision_score, accuracy_score, f1_score

    from networkx.drawing.nx_agraph import graphviz_layout

    jvm.start()

    json_results = {
        'runs': {
            '1': dict()
        }
    }

    try:
        for dataset in os.listdir(datasets_path):
            dataset_name = dataset.split('.')[0]

            json_results['runs']['1'][dataset_name] = dict()

            loader = Loader(classname="weka.core.converters.ArffLoader")

            y_pred_all = []
            y_true_all = []
            heights = []
            n_nodes = []

            for n_fold in it.count():
                try:
                    train_s = loader.load_file(
                        os.path.join(intermediary_path, '%s_fold_%d_train.arff' % (dataset_name, n_fold)))
                    val_s = loader.load_file(
                        os.path.join(intermediary_path, '%s_fold_%d_val.arff' % (dataset_name, n_fold)))
                    test_s = loader.load_file(
                        os.path.join(intermediary_path, '%s_fold_%d_test.arff' % (dataset_name, n_fold)))

                    train_s.relationname = dataset_name
                    val_s.relationname = dataset_name
                    test_s.relationname = dataset_name

                    train_s.class_is_last()
                    val_s.class_is_last()
                    test_s.class_is_last()

                    warnings.warn('WARNING: appending validation set in training set.')
                    for inst in val_s:
                        train_s.add_instance(inst)

                    cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.25", "-M", "2"])
                    # cls = Classifier(classname="weka.classifiers.trees.REPTree",
                    # options=["-M", "2", "-V", "0.001", "-N", "3", "-S", "1", "-L", "-1", "-I", "0.0"])
                    cls.build_classifier(train_s)

                    warnings.warn('WARNING: will only work for binary splits!')
                    graph = cls.graph.encode('ascii')
                    out = StringIO.StringIO(graph)
                    G = nx.Graph(nx.nx_pydot.read_dot(out))

                    # TODO plotting!
                    # fig = plt.figure(figsize=(40, 30))
                    # pos = graphviz_layout(G, root='N0', prog='dot')
                    #
                    # edgelist = G.edges(data=True)
                    # nodelist = G.nodes(data=True)
                    #
                    # edge_labels = {(x1, x2): v['label'] for x1, x2, v in edgelist}
                    # node_colors = {node_id: ('#98FB98' if 'shape' in _dict else '#0099FF') for node_id, _dict in nodelist}
                    # node_colors['N0'] = '#FFFFFF'
                    # node_colors = node_colors.values()
                    #
                    # nx.draw_networkx_nodes(G, pos, node_color=node_colors)
                    # nx.draw_networkx_edges(G, pos, style='dashed', arrows=False)
                    # nx.draw_networkx_labels(G, pos, {k: v['label'] for k, v in G.node.iteritems()})
                    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                    # plt.axis('off')
                    # plt.show()
                    # exit(0)
                    # TODO plotting!

                    heights += [max(map(len, nx.shortest_path(G, source='N0').itervalues()))]
                    n_nodes += [len(G.node)]

                    y_test_true = []
                    y_test_pred = []

                    # y_train_true = []
                    # y_train_pred = []

                    # for index, inst in enumerate(train_s):
                    #     y_train_true += [inst.get_value(inst.class_index)]
                    #     y_train_pred += [cls.classify_instance(inst)]

                    for index, inst in enumerate(test_s):
                        y_test_true += [inst.get_value(inst.class_index)]
                        y_test_pred += [cls.classify_instance(inst)]

                    y_true_all += y_test_true
                    y_pred_all += y_test_pred

                except Exception as e:
                    break

            json_results['runs']['1'][dataset_name] = {
                'confusion_matrix': confusion_matrix(y_true_all, y_pred_all).tolist(),
                'height': heights,
                'n_nodes': n_nodes,
            }

        # interprets
        json_results = json.load(open('/home/henry/Desktop/j48/j48_results.json', 'r'))

        n_runs = len(json_results['runs'].keys())
        some_run = json_results['runs'].keys()[0]
        n_datasets = len(json_results['runs'][some_run].keys())

        df = pd.DataFrame(
            columns=['run', 'dataset', 'test_acc', 'height mean', 'height std', 'n_nodes mean', 'n_nodes std'],
            index=np.arange(n_runs * n_datasets),
            dtype=np.float32
        )

        df['dataset'] = df['dataset'].astype(np.object)

        count_row = 0
        for n_run, run in json_results['runs'].iteritems():
            for dataset_name, dataset in run.iteritems():
                conf_matrix = np.array(dataset['confusion_matrix'], dtype=np.float32)

                test_acc = np.diag(conf_matrix).sum() / conf_matrix.sum()

                height_mean = np.mean(dataset['height'])
                height_std = np.std(dataset['height'])
                n_nodes_mean = np.mean(dataset['n_nodes'])
                n_nodes_std = np.std(dataset['n_nodes'])

                df.loc[count_row] = [
                    int(n_run), str(dataset_name), float(test_acc),
                    float(height_mean), float(height_std), float(n_nodes_mean), float(n_nodes_std)
                ]
                count_row += 1

        print df
        json.dump(json_results, open('j48_results.json', 'w'), indent=2)
        df.to_csv('j48_results.csv', sep=',', quotechar='\"', index=False)

    finally:
        jvm.stop()


def evaluate_ardennes(datasets_path, config_file, output_path, validation_mode='cross-validation'):
    datasets = os.listdir(datasets_path)
    np.random.shuffle(datasets)  # everyday I'm shuffling

    print 'configuration file:'
    print config_file
    config_file['verbose'] = False

    n_runs = config_file['n_runs']

    # --------------------------------------------------- #
    # begin of {removes previous results, create folders}
    # --------------------------------------------------- #
    for i, dataset in enumerate(datasets):
        dataset_name = dataset.split('.')[0]

        if output_path is not None:
            dataset_output_path = os.path.join(output_path, dataset_name)

            if not os.path.exists(dataset_output_path):
                os.mkdir(dataset_output_path)
            else:
                shutil.rmtree(dataset_output_path)
                os.mkdir(dataset_output_path)
    # --------------------------------------------------- #
    # end of {removes previous results, create folders}
    # --------------------------------------------------- #

    dict_results = {'runs': dict()}

    for n_run in xrange(n_runs):
        dict_results['runs'][str(n_run)] = dict()

        for i, dataset in enumerate(datasets):
            dataset_name = dataset.split('.')[0]
            config_file['dataset_path'] = os.path.join(datasets_path, dataset)

            dataset_output_path = os.path.join(output_path, dataset_name)

            if output_path is not None:
                config_file['output_path'] = dataset_output_path
            try:
                dt_dict = __train__(
                    kwargs=config_file,
                    evaluation_mode=validation_mode,
                    n_run=n_run
                )

                dict_results['runs'][str(n_run)][dataset_name] = dt_dict

                json.dump(dict_results, open(os.path.join(output_path, 'results.json'), 'w'), indent=2)
            except Exception as e:
                import warnings
                warnings.warn('Exception found when running %s!' % dataset)
                print(e.message, e.args)


def crunch_result_file(results_path):
    results_file = json.load(
        open(results_path, 'r')
    )

    n_runs = len(results_file['runs'].keys())
    some_run = results_file['runs'].keys()[0]
    n_datasets = len(results_file['runs'][some_run].keys())

    df = pd.DataFrame(
        columns=['run', 'dataset', 'test_acc', 'height', 'n_nodes'],
        index=np.arange(n_runs * n_datasets),
        dtype=np.object
    )

    dtypes = dict(
        run=np.float32, dataset=np.object, test_acc=np.float32,
        height=np.float32, n_nodes=np.float32
    )

    for k, v in dtypes.iteritems():
        df[k] = df[k].astype(v)

    count_row = 0
    for n_run, run in results_file['runs'].iteritems():
        for dataset_name, dataset in run.iteritems():
                conf_matrix = np.array(dataset['confusion_matrix'], dtype=np.float32)

                test_acc = np.diag(conf_matrix).sum() / conf_matrix.sum()

                height_mean = np.mean(dataset['height'])
                n_nodes_mean = np.mean(dataset['n_nodes'])

                df.loc[count_row] = [
                    int(n_run), str(dataset_name), float(test_acc),
                    float(height_mean), float(n_nodes_mean)
                ]
                count_row += 1

    print df

    grouped = df.groupby(by=['dataset'])['test_acc', 'height', 'n_nodes']
    final = grouped.aggregate([np.mean, np.std])

    print final

    final.to_csv(results_path.split('.')[0] + '.csv', sep=',', quotechar='\"')


def crunch_evolution_data(path_results, criteria):
    df = pd.read_csv(path_results)
    for criterion in criteria:
        df.boxplot(column=criterion, by='iteration')
        plt.savefig(path_results.split('.')[0] + '_%s.pdf' % criterion, bbox_inches='tight', format='pdf')
        plt.close()


def generation_statistics(path_results):
    df = pd.read_csv(path_results)
    gb = df.groupby(by='iteration')
    meta = gb.agg([np.min, np.max, np.median, np.mean, np.std])
    meta.to_csv('iteration_statistics.csv')


def crunch_graphical_model(pgm_path, path_datasets):
    from networkx.drawing.nx_agraph import graphviz_layout
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

    dataset = load_dataframe(os.path.join(path_datasets, dataset_name + '.arff'))
    columns = dataset.columns
    n_columns = dataset.shape[1]
    del dataset

    data = []

    with open(pgm_path, 'r') as f:
        csv_w = csv.reader(f, delimiter=',', quotechar='\"')
        for generation, line in enumerate(csv_w):
            series = np.array(line, dtype=np.float).reshape(n_columns, -1)  # each row is an attribute, each column a generation

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


def __run__(train_df, test_df=None, **kwargs):
    t1 = dt.now()

    if 'output_path' in kwargs and kwargs['output_path'] is not None:
        dbhandler = DatabaseHandler(
            path=os.path.join(kwargs['output_path'], kwargs['dataset_name'] + kwargs['mode'].join("  ") + str(t1) + '.db'),
            dataset_name=kwargs['dataset_name'],
            mode=kwargs['mode'],
            n_runs=kwargs['n_runs'],
            n_individuals=kwargs['n_individuals'],
            n_iterations=kwargs['n_iterations'],
            tree_height=kwargs['tree_height'],
            decile=kwargs['decile'],
            random_state=kwargs['random_state']
        )
        dbhandler.set_run(kwargs['run'])
        dbhandler.write_attributes(train_df.columns)
    else:
        dbhandler = None

    inst = Ardennes(
        n_individuals=kwargs['n_individuals'],
        max_height=kwargs['tree_height'],
        n_iterations=kwargs['n_iterations']
    )

    inst.fit(
        train_df=train_df,
        decile=kwargs['decile'],
        test_df=test_df,  # kwargs
        verbose=kwargs['verbose'],  # kwargs
        random_state=kwargs['random_state'],  # kwargs
        dbhandler=dbhandler  # kwargs
    )

    ind = inst.predictor
    y_test_pred = inst.predict(test_df[test_df.columns[:-1]])
    y_test_true = test_df[test_df.columns[-1]]

    test_acc_score = accuracy_score(y_test_true, y_test_pred)

    t2 = dt.now()

    print 'Test acc: %02.2f Height: %d n_nodes: %d Time: %02.2f secs' % (
        test_acc_score, ind.height, ind.n_nodes, (t2 - t1).total_seconds()
    )
    if dbhandler is not None:
        dbhandler.close()

    if 'dict_manager' in kwargs:
        train_hash = DatabaseHandler.get_hash(train_df)
        hashdb = hash((train_hash, kwargs['run']))
        kwargs['dict_manager'][hashdb] = dbhandler

    return test_acc_score


def __train__(dataset_path, tree_height, random_state=None, n_runs=10, n_jobs=8, output_path=None, **kwargs):
    def running(_processes):
        """
        Gets the number of running processes.

        :param _processes: List of processes.
        :return: Number of running processes.
        """
        _sum = 0
        for _process in _processes:
            _sum += int(_process.is_alive())
        return _sum

    def block(_processes, _n_jobs):
        """
        Prevents a new thread from being initialized until a free slot is made available.

        :param _processes: Array of processes, whether running or not.
        :param _n_jobs: Maximum number of concurrent processes.
        """

        while running(_processes) >= _n_jobs:
            time.sleep(1)

    def get_dataset_name(_dataset_path):
        if os.name == 'nt':  # if on windows
            sep = '\\'
        else:  # else on linux, mac
            sep = '/'
        _dataset_name = dataset_path.split(sep)[-1]
        return _dataset_name

    dataset_name = get_dataset_name(dataset_path)

    files = [
        f.replace(dataset_name + '_', '').replace('.arff', '')
        for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))
    ]  # list only files in the given folder

    print 'training ardennes for dataset %s' % dataset_name

    mode = None
    partial_dbs = []

    if 'train' not in files or 'test' not in files:
        mode = 'cross-validation'

        # list of folds, as arff files
        arffs = [load_arff(os.path.join(dataset_path, dataset_name + '_' + f + '.arff')) for f in files]
        dfs = [  # list of folds, as pandas dataframes
            load_dataframe(
                f,
                index=pd.RangeIndex(
                    i * len(f['data']),  # start index
                    (i + 1) * len(f['data']),  # stop index
                    1  # step
                )
            )
            for i, f in enumerate(arffs)
        ]
        full_df = reduce(lambda x, y: x.append(y), dfs)  # type: pd.DataFrame

        for run in xrange(n_runs):
            manager = Manager()
            dict_manager = manager.dict()

            processes = []

            for fold in dfs:
                train_df = full_df.drop(fold.index).reset_index(drop=True)  # type: pd.DataFrame

                p = Process(
                    target=__run__, kwargs=dict(
                        train_df=train_df,
                        test_df=fold.reset_index(drop=True),
                        dataset_name=dataset_name,
                        mode=mode,
                        n_runs=n_runs,
                        n_individuals=kwargs['n_individuals'],
                        n_iterations=kwargs['n_iterations'],
                        tree_height=tree_height,
                        decile=kwargs['decile'],
                        run=run,
                        verbose=kwargs['verbose'],
                        output_path=output_path,
                        random_state=random_state,
                        dict_manager=dict_manager,
                    )
                )

                block(processes, n_jobs)
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            # TODO within runs scope!

            dict_dbs = dict(dict_manager)
            partial_dbs += dict_dbs.values()

    else:  # for training with train and test folds
        # TODO treat in terms of runs!!!
        # TODO treat in terms of runs!!!
        # TODO treat in terms of runs!!!
        mode = 'holdout'

        train_arff = load_arff(os.path.join(dataset_path, dataset_name + '_train' + '.arff'))
        test_arff = load_arff(os.path.join(dataset_path, dataset_name + '_test' + '.arff'))

        train_df = load_dataframe(train_arff)
        test_df = load_dataframe(test_arff)

        dict_manager = dict()

        for run in xrange(n_runs):
            __run__(
                train_df=train_df,
                test_df=test_df,
                dataset_name=dataset_name,
                mode=mode,
                n_runs=n_runs,
                n_individuals=kwargs['n_individuals'],
                n_iterations=kwargs['n_iterations'],
                tree_height=tree_height,
                decile=kwargs['decile'],
                run=run,
                verbose=kwargs['verbose'],
                output_path=output_path,
                random_state=random_state,
                dict_manager=dict_manager
            )

        partial_dbs += dict_manager.values()

    # from now on, all databases were already created
    dball = DatabaseHandler(
        path=os.path.join(output_path, dataset_name + mode.join("  ") + str(dt.now()) + '.db'),
        dataset_name=dataset_name,
        mode=mode,
        n_runs=n_runs,
        n_individuals=kwargs['n_individuals'],
        n_iterations=kwargs['n_iterations'],
        tree_height=tree_height,
        decile=kwargs['decile'],
        random_state=random_state
    )

    for db in partial_dbs:
        dball.union(db)
        os.remove(db.path)

    dball.plot_population()
    dball.close()
