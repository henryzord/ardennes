# coding=utf-8

"""
Performs tests in an 'industrial' fashion.
"""

import json
import os
from main import do_train
from multiprocessing import Process

__author__ = 'Henry Cagnini'


def evaluate_several(datasets_path, output_path, validation_mode='cross-validation', n_jobs=2):
    datasets = os.listdir(datasets_path)

    config_file = json.load(open('config.json', 'r'))

    processes = []

    for i, dataset in enumerate(datasets):
        if ((i % n_jobs) == 0) and i > 0:
            for process in processes:
                process.join()
            processes = []

        config_file['dataset_path'] = os.path.join(datasets_path, dataset)

        p = Process(
            target=do_train, kwargs={
                'config_file': config_file,
                'output_path': output_path,
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
    _output_path = 'metadata'
    validation_mode = 'cross-validation'

    evaluate_several(
        datasets_path=_datasets_path,
        output_path=_output_path,
        validation_mode=validation_mode
    )
