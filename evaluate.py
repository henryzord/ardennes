# coding=utf-8

"""
Performs tests in an 'industrial' fashion.
"""

import json
import os
import itertools as it
import shutil

import pandas as pd
import arff

from main import do_train
import numpy as np
import copy

__author__ = 'Henry Cagnini'

# liver-disordersfold0_train.arff


def __clean_macros__(macros):
    for sets in macros.itervalues():
        os.remove(sets['train'])
        os.remove(sets['test'])
        if 'val' in sets:
            os.remove(sets['val'])


def get_dataset_sets(dataset_name, datasets_path, fold_file, output_path='.', merge_val=True):
    """
    Given a dataset (in .arff format), returns its sets (train, val and test).

    :param dataset_name: The name of the dataset (i.e, 'iris' -- do not pass it with a file extension, as in 'iris.arff'
    :param datasets_path: The path in which the dataset is. It is assumed that is a .arff file.
    :param fold_file: A dictionary with the following structure:
        { \n
            \t'0':  # index of the current fold \n
                \t\t'train': [0, 1, 2, ...]  # index of the training instances \n
                \t\t'test': [500, 501, 502, ...]  # index of the test instances \n
                \t\t'val': [600, 601, 602 ...]  # index of the validation instances \n
            \t'1':  \n
                \t\t... \n
        } \n
    :param output_path: optional - File to write csv, csv dataset (one per fold). Defaults to workpath.
    :param merge_val: optional - Whether to merge training and validation sets. In this case, will not output a validation
        csv dataset.
    :return: A dictionary with the same structure as fold_file, except that it contains (relative) file paths to csv
        datasets.
    """
    arff_dtst = arff.load(open(os.path.join(datasets_path, dataset_name + '.arff'), 'r'))

    macros = dict()

    for n_fold, folds_sets in fold_file.iteritems():
        n_fold = int(n_fold)
        macros[n_fold] = dict()

        macro_train = os.path.join(output_path, '%s_fold_%d_train.csv') % (dataset_name, int(n_fold))
        macro_test = os.path.join(output_path, '%s_fold_%d_test.csv') % (dataset_name, int(n_fold))

        attributes = [x[0] for x in arff_dtst['attributes']]
        np_train_s = pd.DataFrame(arff_dtst['data'], columns=attributes)
        np_test_s = pd.DataFrame(arff_dtst['data'], columns=attributes)

        np_test_s = np_test_s.loc[folds_sets['test']]  # type: pd.DataFrame
        if merge_val:
            np_train_s = np_train_s.loc[folds_sets['train'] + folds_sets['val']]  # type: pd.DataFrame
        else:
            macro_val = os.path.join(output_path, '%s_fold_%d_val.csv') % (dataset_name, int(n_fold))
            np_train_s = np_train_s.loc[folds_sets['train']]
            np_val_s = pd.DataFrame(arff_dtst['data'], columns=attributes).loc[folds_sets['val']]
            np_val_s = np_val_s.sort_values(by=np_val_s.columns[-1])
            np_val_s.to_csv(macro_val, index=False)
            macros[n_fold]['val'] = macro_val

        np_train_s = np_train_s.sort_values(by=np_train_s.columns[-1])
        np_test_s = np_test_s.sort_values(by=np_test_s.columns[-1])

        np_train_s.to_csv(macro_train, index=False)
        np_test_s.to_csv(macro_test, index=False)

        macros[n_fold]['train'] = macro_train
        macros[n_fold]['test'] = macro_test

    return macros


def __generate_intermediary_datasets__(datasets_path, folds_path, output_path):
    import weka.core.jvm as jvm
    from weka.core.converters import Loader
    from weka.core.converters import Saver

    jvm.start()

    for dataset in os.listdir(datasets_path):
        dataset_name = dataset.split('.')[0]
        fold_file = json.load(open(os.path.join(folds_path, dataset_name + '.json'), 'r'))

        csv_loader = Loader(classname="weka.core.converters.CSVLoader")
        arff_saver = Saver(classname='weka.core.converters.ArffSaver')

        macros = get_dataset_sets(
            dataset_name=dataset_name,
            datasets_path=datasets_path,
            fold_file=fold_file,
            output_path=output_path,
            merge_val=False
        )

        for n_fold, folds_sets in macros.iteritems():
            train_s = csv_loader.load_file(macros[n_fold]['train'])
            test_s = csv_loader.load_file(macros[n_fold]['test'])
            val_s = csv_loader.load_file(macros[n_fold]['val'])

            train_s.relationname = dataset_name
            test_s.relationname = dataset_name
            val_s.relationname = dataset_name

            train_s.class_is_last()
            test_s.class_is_last()
            val_s.class_is_last()

            cpy_train = copy.deepcopy(macros[n_fold]['train']).replace('.csv', '.arff')
            cpy_test = copy.deepcopy(macros[n_fold]['test']).replace('.csv', '.arff')
            cpy_val = copy.deepcopy(macros[n_fold]['val']).replace('.csv', '.arff')

            arff_saver.save_file(train_s, cpy_train)
            arff_saver.save_file(test_s, cpy_test)
            arff_saver.save_file(val_s, cpy_val)

            # TODO move csv to output_file!

    jvm.stop()


def evaluate_j48(datasets_path, folds_path):
    import weka.core.jvm as jvm
    from weka.core.converters import Loader
    from weka.classifiers import Classifier

    jvm.start()

    results = dict()

    try:
        for dataset in os.listdir(datasets_path):
            dataset_name = dataset.split('.')[0]

            results[dataset_name] = dict()

            print 'doing for dataset %s' % dataset_name

            fold_file = json.load(open(os.path.join(folds_path, dataset_name + '.json'), 'r'))

            loader = Loader(classname="weka.core.converters.CSVLoader")

            macros = get_dataset_sets(
                dataset_name=dataset_name,
                datasets_path=datasets_path,
                fold_file=fold_file,
                output_path='.',
                merge_val=True
            )

            for n_fold, folds_sets in macros.iteritems():
                train_s = loader.load_file(macros[n_fold]['train'])
                test_s = loader.load_file(macros[n_fold]['test'])

                train_s.relationname = dataset_name
                test_s.relationname = dataset_name

                train_s.class_is_last()
                test_s.class_is_last()

                cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.25", "-M", "2"])
                cls.build_classifier(train_s)

                acc = 0.
                for index, inst in enumerate(test_s):
                    pred = cls.classify_instance(inst)
                    real = inst.get_value(inst.class_index)
                    acc += (pred == real)

                acc /= float(test_s.num_instances)

                results[dataset_name][n_fold] = acc

                print 'dataset %s %d-th fold accuracy: %02.2f' % (dataset_name, int(n_fold), acc)

            __clean_macros__(macros)

        json.dump(results, open('j48_results.json', 'w'), indent=2)

    finally:
        jvm.stop()


def evaluate_ardennes(datasets_path, output_path, validation_mode='cross-validation'):
    datasets = os.listdir(datasets_path)
    np.random.shuffle(datasets)  # everyday I'm shuffling

    config_file = json.load(open('config.json', 'r'))

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
    _datasets_path = 'datasets/numerical'
    _folds_path = 'datasets/folds'
    _output_path = 'metadata'
    _validation_mode = 'cross-validation'

    evaluate_ardennes(
        datasets_path=_datasets_path,
        output_path=_output_path,
        validation_mode=_validation_mode
    )

    # evaluate_j48(_datasets_path, _folds_path)

    # __generate_intermediary_datasets__(_datasets_path, _folds_path, output_path='intermediary')
