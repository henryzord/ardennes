import json

# noinspection PyUnresolvedReferences
from evaluate import evaluate_ardennes, evaluate_j48, \
    crunch_graphical_model,  crunch_result_file, \
    __train__, crunch_evolution_data


if __name__ == '__main__':
    _config_file = json.load(open('config.json', 'r'))
    _datasets_path = 'datasets/numerical'

    _folds_path = 'datasets/folds'
    _output_path = 'metadata'
    _validation_mode = 'cross-validation'
    _intermediary_sets = 'intermediary'

    # # --------------------------------------------------- #
    # crunch_result_file('/home/henry/Desktop/ardennes/r2 [8 runs]/results.json')
    # # --------------------------------------------------- #
    # evaluate_j48(_datasets_path, _intermediary_sets)
    # # --------------------------------------------------- #
    # crunch_graphical_model(
    #     '/home/henry/Desktop/floats/iris_pgm_fold_000_run_004.csv',
    #     _datasets_path
    # )
    # # --------------------------------------------------- #
    # crunch_evolution_data('/home/henry/Desktop/floats/iris_evo_fold_000_run_004.csv')
    # # --------------------------------------------------- #
    # generation_statistics('/home/henry/Desktop/iris_evo_fold_002_run_000.csv')
    # # --------------------------------------------------- #
    # evaluate_ardennes(
    #     datasets_path=_datasets_path,
    #     config_file=_config_file,
    #     output_path=_output_path,
    #     validation_mode=_validation_mode
    # )
    # # --------------------------------------------------- #
    __train__(
        # dataset_path='datasets/gene expression/breastCancer',
        # dataset_path='datasets/numerical/ionosphere',
        dataset_path='datasets/numerical/iris',
        **_config_file
    )
