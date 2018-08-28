import json
import numpy as np

from reporter import BaselineReporter
from treelib import Ardennes
from utils import get_dataset_name, __get_fold__


def ardennes(dataset_path, output_path, params_path, n_fold, n_run):
    params = json.load(open(params_path))

    dataset_name = get_dataset_name(dataset_path)

    full_df, train_index, val_index, test_index = __get_fold__(params=params, dataset_path=dataset_path, n_fold=n_fold)

    n_classes = len(np.unique(full_df))

    X = full_df[full_df.columns[:-1]]
    y = full_df[full_df.columns[-1]]

    X_train = X.loc[train_index]
    X_val = X.loc[val_index]
    X_test = X.loc[test_index]

    y_train = y.loc[train_index]
    y_val = y.loc[val_index]
    y_test = y.loc[test_index]

    reporter = BaselineReporter(
        Xs=[X_train, X_val, X_test],
        ys=[y_train, y_val, y_test],
        n_classes=n_classes,
        set_names=['train', 'val', 'test'],
        dataset_name=dataset_name,
        n_fold=n_fold,
        n_run=n_run,
        output_path=output_path,
        algorithm=Ardennes
    )

    model = Ardennes(
        n_individuals=params['n_individuals'],
        n_generations=params['n_generations'],
        max_height=params['max_height'],
        decile=params['decile'],
        reporter=reporter,
    )

    model = model.fit(
        full_df=full_df,
        train_index=train_index,
        val_index=val_index
    )

    return model
