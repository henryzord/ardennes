import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from reporter import EDAReporter
from treelib import Ardennes
from utils import get_dataset_name, __get_fold__


def ardennes(dataset_path, output_path, params_path, n_fold, n_run):
    params = json.load(open(params_path))

    dataset_name = get_dataset_name(dataset_path)

    df, rest_index, test_index = __get_fold__(params=params, dataset_path=dataset_path, n_fold=n_fold)

    X_test = df.loc[test_index, df.columns[:-1]]
    y_test = df.loc[test_index, df.columns[-1]]
    del test_index  # deletes it to prevent from being using later

    rest_df = df.loc[rest_index]  # type: pd.DataFrame
    rest_df.reset_index(inplace=True, drop=True)

    rest_y = rest_df[rest_df.columns[-1]]
    frac = (float(params['n_folds']) - 2) / (float(params['n_folds']) - 1)

    train_index, val_index = train_test_split(
        rest_df.index, train_size=frac,
        shuffle=True, random_state=params['random_state'], stratify=rest_y
    )

    n_classes = len(np.unique(rest_df))

    X_train = rest_df.loc[train_index, rest_df.columns[:-1]]
    y_train = rest_df.loc[train_index, rest_df.columns[-1]]

    X_val = rest_df.loc[val_index, rest_df.columns[:-1]]
    y_val = rest_df.loc[val_index, rest_df.columns[-1]]

    train_index_bool = np.zeros(len(rest_y), dtype=np.bool)
    train_index_bool[train_index] = True
    val_index_bool = np.zeros(len(rest_y), dtype=np.bool)
    val_index_bool[val_index] = True

    reporter = EDAReporter(
        Xs=[X_train, X_val],
        ys=[y_train, y_val],
        n_classes=n_classes,
        set_names=['train', 'val'],
        dataset_name=dataset_name,
        n_fold=n_fold,
        n_run=n_run,
        output_path=output_path,
    )

    model = Ardennes(
        n_individuals=params['n_individuals'],
        n_generations=params['n_generations'],
        max_height=params['max_height'],
        decile=params['decile'],
        reporter=reporter,
    )

    model = model.fit(
        full_df=rest_df,
        train_index=train_index_bool,
        val_index=val_index_bool
    )

    return model
