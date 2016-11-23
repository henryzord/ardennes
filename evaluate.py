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

    for i, dataset in enumerate(datasets):
        config_file['dataset_path'] = os.path.join(datasets_path, dataset)

        do_train(
            config_file=config_file,
            output_path=output_path,
            evaluation_mode=validation_mode
        )


if __name__ == '__main__':
    _datasets_path = 'datasets/numerical'
    _output_path = 'metadata'
    validation_mode = 'cross-validation'

    evaluate_several(
        datasets_path=_datasets_path,
        output_path=_output_path,
        validation_mode=validation_mode,
        n_jobs=2
    )
