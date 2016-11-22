# coding=utf-8

"""
Performs tests in an 'industrial' fashion.
"""

import json
import os
from main import do_train
from multiprocessing import Process

__author__ = 'Henry Cagnini'


def evaluate_several(datasets_path, folds_path, output_path, n_jobs=2):
    datasets = os.listdir(datasets_path)

    validation_mode = 'cross-validation'

    config_file = json.load(open('config.json', 'r'))

    processes = []

    for i, dataset in enumerate(datasets):
        dataset_name = dataset.split('.')[0]

        if ((i % n_jobs) == 0) and i > 0:
            for process in processes:
                process.join()
            processes = []

        output_folder = os.path.join(output_path, dataset_name)
        config_file['dataset_path'] = os.path.join(datasets_path, dataset)

        p = Process(
            target=do_train, kwargs={
                'config_file': config_file,
                'output_folder': output_folder,
                'evaluation_mode': validation_mode
            }
        )
        p.start()
        processes.append(p)

        import warnings
        warnings.warn('WARNING: testing for only one dataset!')
        p.join()
        exit(0)


if __name__ == '__main__':
    _datasets_path = 'datasets/numerical'
    _folds_path = 'datasets/folds'
    _output_path = 'metadata/first_run'

    evaluate_several(_datasets_path, _folds_path, _output_path)
