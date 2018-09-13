from __future__ import print_function

import argparse
import json
import os

from main import train_ardennes
from reporter import BaselineReporter
from utils import get_dataset_name


def preliminaries(dataset_path, output_path, params_path, n_fold, n_run):
    dataset_name = get_dataset_name(dataset_path)

    # try:
    train_ardennes(
        dataset_path=dataset_path,
        output_path=output_path,
        params_path=params_path,
        n_fold=n_fold,
        n_run=n_run
    )

    # TODO reactivate later
    # except Exception as e:
    #     name = EDAReporter.get_output_file_name(
    #         output_path=output_path,
    #         dataset_name=dataset_name,
    #         n_fold=n_fold, n_run=n_run,
    #         reason='exception'
    #     )
    #
    #     with open(name, 'w') as f:
    #         f.write(str(e) + '\n' + str(e.args))


def test_ardennes(dataset_path, output_path, params_path, n_runs=10, n_jobs=8):
    params = json.load(open(params_path, 'r'))

    if os.path.isfile(dataset_path):  # one dataset
        datasets = [dataset_path.split(os.sep)[-1]]
        dataset_path = os.sep.join(dataset_path.split(os.sep)[:-1])
    else:
        files = os.listdir(dataset_path)  # several datasets; check for arff files
        datasets = [dataset for dataset in files if '.arff' in dataset]

    jobs = []

    for dataset in datasets:
        dataset_name = dataset.split('.')[0]

        print('testing %s dataset' % dataset_name)

        for n_fold in range(params['n_folds']):
            for n_run in range(n_runs):
                print('# --- dataset: %r n_fold: %r/%r n_run: %r/%r --- #' % (
                    dataset_name, n_fold + 1, params['n_folds'], n_run + 1, n_runs
                ))

                preliminaries(
                    dataset_path=os.path.join(dataset_path, dataset),  # TODO solve!
                    output_path=output_path,
                    params_path=params_path,
                    n_fold=n_fold,
                    n_run=n_run,
                )

                # TODO reactivate
                # while len(jobs) >= n_jobs:
                #     jobs = __get_running_processes__(jobs)
                #     time.sleep(5)

                # job = Process(
                #     target=preliminaries,
                #     kwargs=dict(
                #         dataset_path=os.path.join(datasets_path, dataset),
                #         output_path=output_path,
                #         params_path=params_path,
                #         n_fold=n_fold,
                #         n_run=n_run,
                #     )
                # )
                # job.start()
                # jobs += [job]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Main script for running Ardennes, '
                    'an Estimation of Estimation of Distribution Algorithm for decision-tree induction.'
    )
    parser.add_argument(
        '-d', action='store', required=True,
        help='Path for datasets (either a folder or a file). If it\'s a folder, will run Ardennes for all datasets'
             '(i.e. files with .arff extension) there. If a file, will run Ardennes only for that dataset.'
    )
    parser.add_argument(
        '-m', action='store', required=True,
        help='Path to metadata folder. The folder must be pre-existent, even if empty.'
    )
    parser.add_argument(
        '-p', action='store', required=True,
        help='Path to Ardennes\' .json parameter file.'
    )
    parser.add_argument(
        '-r', action='store', required=True,
        help='Path to results .csv file that will be created with all execution results.'
    )
    parser.add_argument(
        '--runs', action='store', default=10, required=False, type=int,
        help='Number of runs for each cross-validation step. Defaults to 10.'
    )
    parser.add_argument(
        '--jobs', action='store', default=4, required=False, type=int,
        help='Number of parallel cross validation steps to run. Defaults to 1. Must not be higher than the number of'
             'cores a computer has.'
    )

    args = parser.parse_args()

    test_ardennes(
        dataset_path=args.d,
        output_path=args.m,
        params_path=args.p,
        n_runs=args.runs,
        n_jobs=args.jobs
    )
    # TODO reactivate
    # BaselineReporter.generate_summary(args.m, args.r)
