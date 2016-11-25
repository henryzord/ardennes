# coding=utf-8

"""
Performs tests in an 'industrial' fashion.
"""

import json
import os
import itertools as it

from main import do_train
import numpy as np

__author__ = 'Henry Cagnini'


def evaluate_j48(datasets_path, folds_path):
    import weka.core.jvm as jvm
    from weka.core.converters import Loader
    from weka.classifiers import Classifier

    dataset_name = 'iris'
    iris_fold = json.load(open(os.path.join(folds_path, dataset_name + '.json'), 'r'))

    jvm.start()

    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(os.path.join(datasets_path, 'iris.arff'))
    data.class_is_last()

    for folds_sets in iris_fold.itervalues():
        test_s = data.copy_instances(data)
        train_s = data.copy_instances(data)

        for name, set in folds_sets.iteritems():
            some_set = None
            if name == 'test':
                some_set = train_s
            elif name == 'train':
                some_set = test_s
            elif name == 'val':
                some_set = test_s

            some_set.delete(set)  # TODO by using this method, does not keep other indices!

        cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.25", "-M", "2"])
        cls.build_classifier(train_s)

        acc = 0.
        for index, inst in enumerate(test_s):
            pred = cls.classify_instance(inst)
            real = inst.get_value(data.class_index)
            acc += (pred == real)

        acc /= float(data.num_instances)

        print 'accuracy: %02.2f' % acc

    jvm.stop()


def evaluate_several(datasets_path, output_path, validation_mode='cross-validation', n_jobs=2):
    datasets = os.listdir(datasets_path)
    np.random.shuffle(datasets)  # everyday I'm shuffling

    config_file = json.load(open('config.json', 'r'))

    for i, dataset in enumerate(datasets):
        config_file['dataset_path'] = os.path.join(datasets_path, dataset)

        try:
            do_train(
                config_file=config_file,
                output_path=output_path,
                evaluation_mode=validation_mode
            )
        except:
            import warnings
            warnings.warn('exception found when running %s!' % dataset)


if __name__ == '__main__':
    _datasets_path = 'datasets/numerical'
    _folds_path = 'datasets/folds'
    _output_path = 'metadata'
    _validation_mode = 'cross-validation'

    # evaluate_several(
    #     datasets_path=_datasets_path,
    #     output_path=_output_path,
    #     validation_mode=_validation_mode,
    #     n_jobs=2
    # )

    evaluate_j48(_datasets_path, _folds_path)
