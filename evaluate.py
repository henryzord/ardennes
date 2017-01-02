# coding=utf-8

"""
Performs tests in an 'industrial' fashion.
"""

import json
import os
import shutil

import itertools as it

import StringIO
import warnings

from main import do_train
import numpy as np

import networkx as nx
import math

__author__ = 'Henry Cagnini'


def __clean_macros__(macros):
    for sets in macros.itervalues():
        os.remove(sets['train'])
        os.remove(sets['test'])
        if 'val' in sets:
            os.remove(sets['val'])


def evaluate_j48(datasets_path, intermediary_path):
    # for examples on how to use this function, refer to
    # http://pythonhosted.org/python-weka-wrapper/examples.html#build-classifier-on-dataset-output-predictions
    import weka.core.jvm as jvm
    from weka.core.converters import Loader
    from weka.classifiers import Classifier

    jvm.start()

    results = {
        'runs': {
            '1': dict()
        }
    }

    try:
        for dataset in os.listdir(datasets_path):
            dataset_name = dataset.split('.')[0]

            results['runs']['1'][dataset_name] = dict(folds=dict())

            loader = Loader(classname="weka.core.converters.ArffLoader")

            for n_fold in it.count():
                try:
                    train_s = loader.load_file(os.path.join(intermediary_path, '%s_fold_%d_train.arff' % (dataset_name, n_fold)))
                    val_s = loader.load_file(os.path.join(intermediary_path, '%s_fold_%d_val.arff' % (dataset_name, n_fold)))
                    test_s = loader.load_file(os.path.join(intermediary_path, '%s_fold_%d_test.arff' % (dataset_name, n_fold)))

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
                    cls.build_classifier(train_s)

                    warnings.warn('WARNING: will only work for binary splits!')
                    graph = cls.graph.encode('ascii')
                    out = StringIO.StringIO(graph)
                    G = nx.Graph(nx.nx_pydot.read_dot(out))
                    n_nodes = G.number_of_nodes()
                    height = math.ceil(np.log2(n_nodes + 1))

                    acc = 0.
                    for index, inst in enumerate(test_s):
                        pred = cls.classify_instance(inst)
                        real = inst.get_value(inst.class_index)
                        acc += (pred == real)

                    acc /= float(test_s.num_instances)

                    results['runs']['1'][dataset_name]['folds'][n_fold] = {
                        'acc': acc,
                        'height': height
                    }

                    print 'dataset %s %d-th fold accuracy: %02.2f tree height: %d' % (dataset_name, int(n_fold), acc, height)

                except Exception as e:
                    accs = np.array(
                        [x['acc'] for x in results['runs']['1'][dataset_name]['folds'].itervalues()]
                    )

                    heights = np.array(
                        [x['height'] for x in results['runs']['1'][dataset_name]['folds'].itervalues()]
                    )

                    print 'dataset %s mean accuracy: %0.2f +- %02.2f tree height: %2.2f +- %2.2f' % (
                        dataset_name, accs.mean(), accs.std(), heights.mean(), heights.std()
                    )
                    break

        json.dump(results, open('j48_results.json', 'w'), indent=2)

    finally:
        jvm.stop()


def evaluate_ardennes(datasets_path, config_file, output_path, validation_mode='cross-validation'):
    datasets = os.listdir(datasets_path)
    np.random.shuffle(datasets)  # everyday I'm shuffling

    print 'configuration file:'
    print config_file

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
                dt_dict = do_train(
                    config_file=config_file,
                    evaluation_mode=validation_mode,
                    n_run=n_run
                )

                dict_results['runs'][str(n_run)][dataset_name] = dt_dict

                json.dump(dict_results, open(os.path.join(output_path, 'results.json'), 'w'), indent=2)
            except Exception as e:
                import warnings
                warnings.warn('Exception found when running %s!' % dataset)
                print(e.message, e.args)

if __name__ == '__main__':
    _datasets_path = 'datasets/temp'
    _folds_path = 'datasets/folds'
    _output_path = 'metadata'
    _validation_mode = 'cross-validation'
    _intermediary_sets = 'intermediary'
    _config_file = json.load(open('config.json', 'r'))

    evaluate_ardennes(
        datasets_path=_datasets_path,
        config_file=_config_file,
        output_path=_output_path,
        validation_mode=_validation_mode
    )

    # evaluate_j48(_datasets_path, _intermediary_sets)
