# coding=utf-8

import os
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris

from genotype import Individual, Node

__author__ = 'Henry Cagnini'


class GM(object):
    _pred_attr = None
    _target_attr = None
    _class_values = None
    
    def __init__(self, pred_attr, target_attr, class_values):
        if any(map(lambda x: x is None, [self._pred_attr, self._target_attr])):
            GM._pred_attr = pred_attr.values.tolist()
            GM._target_attr = target_attr
            GM._class_values = class_values
        
        self._pred_attr = GM._pred_attr
        self._target_attr = GM._target_attr
        self._class_values = GM._class_values
    
    def sample(self, id_node):
        pass


class StartGM(GM):
    """
    Initial PMF for generating diverse individuals.
    """
    
    def __init__(self, pred_attr, target_attr, class_values, target_add):
        super(StartGM, self).__init__(pred_attr, target_attr, class_values)
        self._target_add = target_add  # type: float
    
    def sample(self, id_node):
        depth = Node.get_depth(id_node)
        
        target_prob = np.clip(depth * self._target_add, a_min=0., a_max=1.)  # type: float
        pred_prob = [(1. - target_prob) / len(self._pred_attr) for x in xrange(len(self._pred_attr))]  # type: list
        a = self._pred_attr + [self._target_attr]  # type: list
        p = pred_prob + [target_prob]  # type: list
        
        chosen = np.random.choice(a=a, p=p)
        return chosen


class FinalGM(GM):
    def __init__(self, pred_attr, target_attr, class_values):
        super(FinalGM, self).__init__(pred_attr, target_attr, class_values)
        self._graph = None
    
    def update(self, fittest_pop):
        graphs = map(lambda x: x.tree, fittest_pop)
        
        max_node = max(
            map(
                lambda x:
                max(x),
                map(
                    lambda x: x.node.keys(),
                    graphs
                )
            )
        )
        
        inner = nx.DiGraph()
        
        for i in xrange(max_node + 1):
            node_labels = []
            for graph in graphs:
                try:
                    node_label = graph.node[i]['label']
                    if node_label in self._class_values:
                        node_label = self._target_attr
                    node_labels += [node_label]
                except KeyError:
                    pass  # individual does not have the presented node
            
            count = Counter(node_labels)
            count_sum = max(sum(count.itervalues()), 1.)  # prevents zero division
            prob = {k: v / float(count_sum) for k, v in count.iteritems()}
            inner.add_node(n=i, attr_dict={k: v for k, v in prob.iteritems()})
            parent = Node.get_parent(i)
            if parent is not None:
                inner.add_edge(parent, i)
        
        self._graph = inner
    
    def sample(self, id_node):
        a, p = zip(*self._graph.node[id_node].items())
        try:
            chosen = np.random.choice(a=a, p=p)
        except ValueError:
            chosen = None  # when the node doesn't have any probability attached to it
        return chosen

    @property
    def graph(self):
        return self._graph

# def set_pmf(pmf, fittest):
#     nodes = np.array(map(lambda x: x.nodes, fittest))
#     counts = map(lambda v: Counter(v), nodes.T)
#
#     n_internal, n_vals = pmf.shape
#     n_fittest = fittest.shape[0]
#
#     for i in xrange(n_internal):  # for each node index
#         for j in xrange(n_vals):
#             try:
#                 pmf[i, j] = counts[i][j] / float(n_fittest)
#             except KeyError:
#                 pmf[i, j] = 0.
#
#     return pmf


def init_pop(n_individuals, pmf, sets):
    # TODO implement with threading.
    
    pop = np.array(
        map(
            lambda x: Individual(
                ind_id=x,
                initial_pmf=pmf,
                sets=sets
            ),
            xrange(n_individuals)
        )
    )
    return pop


def get_folds(df, n_folds=10, random_state=None):
    from sklearn.cross_validation import StratifiedKFold
    
    Y = df[df.columns[-1]]
    
    folds = StratifiedKFold(Y, n_folds=n_folds, shuffle=True, random_state=random_state)
    return folds


def early_stop(gm, diff=0.01):
    """
    
    :type gm: FinalGM
    :param gm: The Probabilistic Graphical Model (GM) for the current generation.
    :type diff: float
    :param diff: Maximum allowed uncertainty for each probability, for each node.
    :return:
    """
    
    def __max_prob__(node):
        if len(node) > 0:
            max_prob = reduce(
                max,
                map(
                    lambda prob: min(abs(1. - prob), prob),
                    node.itervalues()
                )
            )
            return max_prob
        return 0.
    
    nodes = gm.graph.node
    maximum_prob = reduce(
        max,
        map(
            __max_prob__,
            nodes.itervalues()
        )
    )
    if maximum_prob < diff:
        return True
    return False


def get_node_count(n_nodes):
    n_leaf = (n_nodes + 1) / 2
    n_internal = n_nodes - n_leaf
    return n_internal, n_leaf


def main_loop(sets, n_individuals, target_add, n_iterations=100, inf_thres=0.9, diff=0.01, verbose=True):
    pred_attr = sets['train'].columns[:-1]
    target_attr = sets['train'].columns[-1]
    class_values = sets['train'][sets['train'].columns[-1]].unique()
    
    # pmf only for initializing the population
    gm = StartGM(pred_attr=pred_attr, target_attr=target_attr, class_values=class_values, target_add=target_add)
    
    population = init_pop(
        n_individuals=n_individuals,
        pmf=gm,
        sets=sets
    )
    
    # changes the pmf to a final one
    gm = FinalGM(pred_attr=pred_attr, target_attr=target_attr, class_values=class_values)
    
    fitness = np.array(map(lambda x: x.fitness, population))
    
    # threshold where individuals will be picked for PMF updatting/replacing
    integer_threshold = int(inf_thres * n_individuals)
    
    n_past = 15
    past = np.random.rand(n_past)
    
    iteration = 0
    while iteration < n_iterations:  # evolutionary process
        mean = np.mean(fitness)  # type: float
        median = np.median(fitness)  # type: float
        _max = np.max(fitness)  # type: float
        
        if verbose:
            print 'mean: %+0.6f\tmedian: %+0.6f\tmax: %+0.6f' % (mean, median, _max)
        
        # TODO slow. test other implementation!
        borderline = np.partition(fitness, integer_threshold)[integer_threshold]
        fittest_pop = population[np.flatnonzero(fitness >= borderline)]  # TODO slow. test other implementation!
        
        gm.update(fittest_pop)
        
        to_replace = population[np.flatnonzero(fitness < borderline)]  # TODO slow. test other implementation!
        for ind in to_replace:
            ind.sample(gm)
        
        if early_stop(gm, diff):
            break
        
        fitness = np.array(map(lambda x: x.fitness, population))
        
        iteration += 1
    
    fittest_ind = population[np.argmax(fitness)]
    return fittest_ind


def get_iris(n_folds=10, random_state=None):
    data = load_iris()
    
    X, Y = data['data'], data['target_names'][data['target']]
    
    df = pd.DataFrame(X, columns=data['feature_names'])
    df['class'] = pd.Series(Y, index=df.index)
    
    folds = get_folds(df, n_folds=n_folds, random_state=random_state)
    
    return df, folds


def get_dataset(path, n_folds=10, random_state=None):
    df = pd.read_csv(path, sep=',')
    folds = get_folds(df, n_folds=n_folds, random_state=random_state)
    return df, folds


# @deprecated
# def get_topdown_metadata(x_train, y_train, x_test, y_test):
#     inst = DecisionTreeClassifier(
#         criterion='entropy',
#         splitter='best',
#         max_depth=None,  # as much depth as required
#         min_samples_leaf=1,  # minimum of 1 instance per leaf
#         max_leaf_nodes=None,  # as much leafs as required
#         presort=True
#     )
#     inst.fit(x_train, y_train)
#
#     depth = inst.tree_.max_depth
#
#     h = inst.predict(x_test)
#     acc = (y_test == h).sum() / float(y_test.shape[0])
#
#     # for testing purposes
#     # export_graphviz(inst)
#
#     return depth, acc


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
    
    for j in xrange(kwargs['n_run']):
        fittest = main_loop(
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

    tree_depth = None
    random_state = 1
    n_folds = 10

    kwargs = {
        'random_state': None,
        'n_individuals': 100,
        'n_run': 10,
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
    
    # df, folds = get_iris(n_folds=n_folds)  # iris dataset
    dataset_path = '/home/henryzord/Projects/forrestTemp/datasets/iris.csv'
    df, folds = get_dataset(dataset_path, n_folds=n_folds)  # csv-stored datasets

    for i, (arg_train, arg_test) in enumerate(folds):
        run_fold(i, df, arg_train, arg_test, **kwargs)
        
        # t = threading.Thread(target=run_fold, args=(i, df, arg_train, arg_test, kwargs))
        # t.daemon = True
        # t.start()
    
# plt.show()


if __name__ == '__main__':
    main()
