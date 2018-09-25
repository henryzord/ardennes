# coding=utf-8

import csv
import itertools as it
import json
import os
import warnings
from io import StringIO

import networkx as nx
import pandas as pd
from sklearn.metrics import confusion_matrix

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

        print(df)
        json.dump(json_results, open('j48_results.json', 'w'), indent=2)
        df.to_csv('j48_results.csv', sep=',', quotechar='\"', index=False)

    finally:
        jvm.stop()


def crunch_graphical_model(pgm_path, path_datasets):
    from networkx.drawing.nx_agraph import graphviz_layout
    import plotly.graph_objs as go
    from plotly.offline import plot

    def build_graph(series):
        G = nx.DiGraph()

        node_labels = dict()

        for node_id in range(series.shape[1]):
            probs = series[:, node_id]

            G.add_node(
                node_id,
                attr_dict=dict(
                    color=max(probs),
                    probs='<br>'.join(['%2.3f : %s' % (y, x) for x, y in zip(columns, probs)])
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

    dataset = path_to_dataframe(os.path.join(path_datasets, dataset_name + '.arff'))
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
