# coding=utf-8

"""
Performs tests in an 'industrial' fashion.
"""

import json
import os
from main import train
from multiprocessing import Process

__author__ = 'Henry Cagnini'

datasets_path = '/home/henry/Projects/ardennes/datasets/numerical'


if __name__ == '__main__':
    datasets = os.listdir(datasets_path)
    output_path = 'metadata/[500 ind, 50 iter, 50 decile]'
    validation_mode = 'cross-validation'
    chunk_size = 2

    config_file = json.load(open('input.json', 'r'))

    processes = []

    for i, dataset in enumerate(datasets):
        if ((i % chunk_size) == 0) and i > 0:
            for process in processes:
                process.join()
            processes = []

        config_file['output_file'] = os.path.join(output_path, dataset.split('.')[0].strip() + '_output.csv')
        config_file['dataset_path'] = os.path.join(datasets_path, dataset)

        p = Process(target=train, args=(config_file, validation_mode))
        p.start()
        processes.append(p)
