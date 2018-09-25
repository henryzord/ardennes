import csv
import itertools as it
import os
import pandas as pd

import numpy as np
import pathlib2
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class BaseReporter(object):
    """
    Base class for reporting results of baseline and EDA algorithms.
    """

    metrics = [
        ('accuracy', accuracy_score),
        ('precision-micro', lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro')),
        ('precision-macro', lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro')),
        ('precision-weighted', lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted')),
        ('recall-micro', lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro')),
        ('recall-macro', lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro')),
        ('recall-weighted', lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted')),
        ('f1-micro', lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro')),
        ('f1-macro', lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')),
        ('f1-weighted', lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')),
    ]

    def __init__(
            self, Xs, ys,
            n_variables, column_names,
            set_names, dataset_name, n_fold, n_run, output_path
    ):
        self.Xs = Xs
        self.ys = ys
        self.set_sizes = list(map(len, self.ys))
        self.set_names = set_names
        self.dataset_name = dataset_name

        self.n_run = n_run
        self.n_fold = n_fold
        self.n_variables = n_variables
        self.column_names = column_names
        self.output_path = output_path

    @staticmethod
    def get_output_file_name(output_path, dataset_name, n_fold, n_run, reason):
        """
        Formats the name of the output file.

        :param output_path: Path to output file (without the file name, obviously).
        :param dataset_name: Name of the dataset to be tested.
        :param n_fold: Current fold being tested.
        :param n_run: Current run being tested.
        :param reason: Any additional information regarding the file, e.g., use pop for population file or gm for
        graphical model.
        :return: The formatted name.
        """

        name = os.path.join(
            output_path,
            '-'.join([dataset_name, str(n_fold), str(n_run), reason]) + '.csv'
        )
        return name

    @staticmethod
    def generate_summary(path_read, path_out):
        """
        Based on metadata collected at test time, generates a single csv file with the summary of results of all
        collected metrics.
        :param path_read: Path where all metadata files are.
        :param path_out: Path to the file (with the file name and csv termination) to output summary.
        """
        pass


class BaselineReporter(BaseReporter):
    """
    A class for reporting the partial results of baseline algorithms.
    """

    def __init__(self, Xs, ys, n_variables, column_names, set_names, dataset_name, n_fold, n_run, output_path, algorithm):
        """
        Initializes a reporter, which will follow the EDA throughout evolution,
            reporting the performance of its population.
        :param Xs: Predictive attributes of subsets (e.g., train, validation, test).
        :param ys: Labels of subsets (e.g., train, validation, test).
        :param n_variables: Number of classes in the problem.
        :param column_names: name of columns in dataset.
        :param set_names: Name of the sets (e.g., train, validation, test).
        :param dataset_name: Name of the tested dataset.
        :param n_fold: Current fold index.
        :param n_run: Current run index.
        :param output_path: Path to output meta-files.
        :param algorithm: the name of the algorithm being tested.
        """

        super(BaselineReporter, self).__init__(
            Xs=Xs,
            ys=ys,
            n_variables=n_variables,
            column_names=column_names,
            set_names=set_names,
            dataset_name=dataset_name,
            n_fold=n_fold,
            n_run=n_run,
            output_path=output_path
        )

        self.population_file = self.get_output_file_name(
            output_path=self.output_path, dataset_name=self.dataset_name,
            n_fold=self.n_fold, n_run=self.n_run,
            reason=algorithm.__name__
        )

        with open(self.population_file, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(
                ['dataset', 'n_fold', 'n_run', 'set_name', 'set_size'] +
                [a for a, b in self.metrics]
            )

    def save_baseline(self, ensemble):

        with open(self.population_file, 'a') as f:
            writer = csv.writer(f, delimiter=',')

            counter = 0
            for set_name, set_size, set_x, set_y in zip(self.set_names, self.set_sizes, self.Xs, self.ys):
                preds = ensemble.predict(set_x)
                results = []
                for metric_name, metric_func in BaseReporter.metrics:
                    results += [metric_func(y_true=set_y, y_pred=preds)]

                writer.writerow(
                    [self.dataset_name, self.n_fold, self.n_run, set_name, set_size] + results
                )
                counter += 1

    @staticmethod
    def generate_summary(path_read, path_out):
        files = [xx for xx in pathlib2.Path(path_read).iterdir() if xx.is_file()]
        files = list(map(lambda x: str(x).split('/')[-1].split('.')[0].split('-'), files))
        summary = pd.DataFrame(files, columns=['dataset_name', 'n_fold', 'n_run', 'algorithm'])
        summary['n_fold'] = summary['n_fold'].astype(np.int32)
        summary['n_run'] = summary['n_run'].astype(np.int32)

        algorithms = summary['algorithm'].unique()
        datasets = summary['dataset_name'].unique()
        n_folds = len(summary['n_fold'].unique())
        n_runs = len(summary['n_run'].unique())

        metric_names = [metric_name for metric_name, metric in BaseReporter.metrics]

        result_df = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(list(it.product(algorithms, datasets))),
            columns=list(
                it.chain(*zip(map(lambda x: x + ' mean', metric_names), map(lambda x: x + ' std', metric_names)))),
            dtype=np.float32
        )

        for algorithm in algorithms:
            for dataset in datasets:
                dataset_size = None

                __local_metrics = {k: dict() for k in metric_names}
                for n_fold in range(n_folds):
                    for n_run in range(n_runs):
                        current = pd.read_csv(
                            os.path.join(
                                path_read,
                                '-'.join([dataset, str(n_fold), str(n_run), algorithm]) + '.csv',
                            ),
                            sep=','
                        )
                        # gets dataset size
                        if dataset_size is None:
                            set_names = current['set_name'].unique()
                            dataset_size = 0.
                            for set_name in set_names:
                                dataset_size += int((current.loc[current['set_name'] == set_name].iloc[0])['set_size'])

                        current = current.loc[current['set_name'] == 'test']
                        for metric_name in metric_names:
                            try:
                                __local_metrics[metric_name][n_run] += float(current[metric_name]) * \
                                                                       (
                                                                       float(current['set_size']) / float(dataset_size))
                            except KeyError:
                                __local_metrics[metric_name][n_run] = float(current[metric_name]) * \
                                                                      (float(current['set_size']) / float(dataset_size))

                metric_means = {k: np.mean(list(v.values())) for k, v in __local_metrics.items()}
                metric_stds = {k: np.std(list(v.values())) for k, v in __local_metrics.items()}

                for (metric_name, metric_mean), (metric_name, metric_std) in \
                        zip(metric_means.items(), metric_stds.items()):
                    result_df.loc[(algorithm, dataset)][metric_name + ' mean'] = metric_mean
                    result_df.loc[(algorithm, dataset)][metric_name + ' std'] = metric_std

        result_df.to_csv(path_out, index=True, sep=',', float_format='%0.8f')


class EDAReporter(BaseReporter):
    """
    A class for reporting the partial results of the Ensemble class.
    """

    def __init__(self, Xs, ys, n_variables, column_names, set_names, dataset_name, n_fold, n_run, output_path):
        """
        Initializes a reporter, which will follow the EDA throughout evolution, reporting the performance of its
            population.
        :param Xs: Predictive attributes of subsets (e.g., train, validation, test).
        :param ys: Labels of subsets (e.g., train, validation, test).
        :param n_variables: Number of variables for the graphical model.
        :param column_names: name of columns in dataset.
        :param set_names: Name of the sets (e.g., train, validation, test).
        :param dataset_name: Name of the tested dataset.
        :param n_fold: Current fold index.
        :param n_run: Current run index.
        :param output_path: Path to output meta-files.
        """

        super(EDAReporter, self).__init__(
            Xs=Xs,
            ys=ys,
            n_variables=n_variables,
            column_names=column_names,
            set_names=set_names,
            dataset_name=dataset_name,
            n_fold=n_fold,
            n_run=n_run,
            output_path=output_path
        )

        self.population_file = self.get_output_file_name(
            output_path=self.output_path, dataset_name=self.dataset_name,
            n_fold=self.n_fold, n_run=self.n_run, reason='pop'
        )
        self.gm_file = self.get_output_file_name(
            output_path=self.output_path, dataset_name=self.dataset_name,
            n_fold=self.n_fold, n_run=self.n_run, reason='gm'
        )

        with open(self.population_file, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(
                ['dataset', 'n_fold', 'n_run', 'generation', 'set_name', 'set_size',
                 'elite', 'height', 'n_nodes', 'fitness', 'quality'] +
                [a for a, b in EDAReporter.metrics]
            )

        with open(self.gm_file, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(
                ['dataset', 'n_fold', 'n_run', 'generation'] +
                ['node_%d_%s' % (d, s) for d, s in it.product(range(n_variables), column_names)]
            )

    def save_population(self, generation, population):
        """
        Saves population metadata to a file. Calculates metrics regarding each individual (for example, accuracy,
            precision, etc).
        """

        with open(self.population_file, 'a') as f:
            writer = csv.writer(f, delimiter=',')

            counter = 0
            for set_name, set_size, set_x, set_y in zip(self.set_names, self.set_sizes, self.Xs, self.ys):
                for i, individual in population.iterrows():
                    preds = individual['P'].predict(set_x)
                    results = []
                    for metric_name, metric_func in EDAReporter.metrics:
                        results += [metric_func(y_true=set_y, y_pred=preds)]

                    writer.writerow(
                        [self.dataset_name, self.n_fold, self.n_run, generation, set_name, set_size] +
                        [individual['A'], individual['P'].height, individual['P'].n_nodes, individual['P'].fitness, individual['P'].quality] +
                        results
                    )
                    counter += 1

    def save_gm(self, generation, gm):
        """
        Saves a probabilistic graphical model to a file.
        """

        with open(self.gm_file, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(
                [self.dataset_name, self.n_fold, self.n_run, generation] +
                gm.p.ravel().tolist()
            )

    @staticmethod
    def generate_summary(path_read, path_out):
        files = [xx for xx in pathlib2.Path(path_read).iterdir() if (xx.is_file() and 'pop.csv' in str(xx))]
        files = list(map(lambda x: str(x).split('/')[-1].split('.')[0].split('-'), files))
        summary = pd.DataFrame(files, columns=['dataset_name', 'n_fold', 'n_run', 'pop'])
        summary['n_fold'] = summary['n_fold'].astype(np.int32)
        summary['n_run'] = summary['n_run'].astype(np.int32)

        datasets = summary['dataset_name'].unique()
        n_folds = len(summary['n_fold'].unique())
        n_runs = len(summary['n_run'].unique())

        metric_names = [metric_name for metric_name, metric in BaseReporter.metrics]

        result_df = pd.DataFrame(
            index=datasets,
            columns=list(
                it.chain(*zip(map(lambda x: x + ' mean', metric_names), map(lambda x: x + ' std', metric_names)))),
            dtype=np.float32
        )

        total_steps = len(datasets) * n_folds * n_runs

        global_counter = 0
        for dataset_name in datasets:
            dataset_size = None

            # checks whether required dataset exists
            partial = summary.loc[summary['dataset_name'] == dataset_name]
            if len(partial.index) != (n_folds * n_runs):
                global_counter += (n_folds * n_runs)
                print ('%04.d/%04.d steps done [skipping %s]' % (
                    global_counter, total_steps, dataset_name
                ))
                continue  # skips

            __local_metrics = {k: dict() for k in metric_names}  # one dictionary for each dataset
            for n_fold in range(n_folds):
                for n_run in range(n_runs):
                    file_name = '-'.join([dataset_name, str(n_fold), str(n_run), 'pop']) + '.csv'

                    current = pd.read_csv(
                        os.path.join(
                            path_read,
                            file_name,
                        ),
                        sep=','
                    )
                    # gets dataset size
                    if dataset_size is None:
                        set_names = current['set_name'].unique()
                        dataset_size = 0.
                        for set_name in set_names:
                            dataset_size += int(
                                (current.loc[current['set_name'] == set_name].iloc[0])['set_size'])

                    current['generation'] = current['generation'].astype(np.int32)
                    current['fitness'] = current['fitness'].astype(np.float32)

                    # gets best individual from last generation
                    current = current.loc[
                        (current['generation'] == current['generation'].max()) & (current['set_name'] == 'test')
                    ]
                    current = current.loc[current['fitness'] == current['fitness'].min()].iloc[0]

                    for metric_name in metric_names:
                        try:
                            __local_metrics[metric_name][n_run] += float(current[metric_name]) * \
                                                                   (float(current['set_size']) / float(
                                                                       dataset_size))
                        except KeyError:
                            __local_metrics[metric_name][n_run] = float(current[metric_name]) * \
                                                                  (float(current['set_size']) / float(
                                                                      dataset_size))

                    global_counter += 1
                    print ('%04.d/%04.d steps done' % (global_counter, total_steps))

            metric_means = {k: np.mean(list(v.values())) for k, v in __local_metrics.items()}
            metric_stds = {k: np.std(list(v.values())) for k, v in __local_metrics.items()}

            for (metric_name, metric_mean), (metric_name, metric_std) in \
                    list(zip(metric_means.items(), metric_stds.items())):
                result_df.loc[dataset_name][metric_name + ' mean'] = metric_mean
                result_df.loc[dataset_name][metric_name + ' std'] = metric_std

        result_df.to_csv(path_out, index=True, sep=',', float_format='%0.8f')
