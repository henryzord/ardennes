import json

# noinspection PyUnresolvedReferences
from evaluate import evaluate_ardennes, evaluate_j48, crunch_graphical_model, \
    grid_optimizer, crunch_parametrization, crunch_result_file, do_train, crunch_evolution_data, \
    generation_statistics, custom_pop_stat

if __name__ == '__main__':
    _config_file = json.load(open('config.json', 'r'))
    _datasets_path = 'datasets/numerical'

    _folds_path = 'datasets/folds'
    _output_path = 'metadata'
    _validation_mode = 'cross-validation'
    _intermediary_sets = 'intermediary'

    # custom_pop_stat('/home/henry/Projects/ardennes/metadata')
    # # --------------------------------------------------- #
    # crunch_result_file('/home/henry/Projects/ardennes/metadata/results.json')
    # # --------------------------------------------------- #
    # evaluate_j48(_datasets_path, _intermediary_sets)
    # # --------------------------------------------------- #
    # crunch_graphical_model(
    #     '/home/henry/Desktop/floats/iris_pgm_fold_000_run_004.csv',
    #     _datasets_path
    # )
    # # --------------------------------------------------- #
    # grid_optimizer(_config_file, _datasets_path, output_path='/home/henry/Desktop/parametrizations')
    # # --------------------------------------------------- #
    # crunch_parametrization('parametrization_hayes-roth-full.csv')
    # # --------------------------------------------------- #
    # _evaluation_mode = 'holdout'
    # do_train(config_file=_config_file, n_run=0, evaluation_mode=_evaluation_mode)
    # --------------------------------------------------- #
    # crunch_evolution_data('/home/henry/Desktop/floats/iris_evo_fold_000_run_004.csv')
    # --------------------------------------------------- #
    # generation_statistics('/home/henry/Desktop/iris_evo_fold_002_run_000.csv')
    # --------------------------------------------------- #
    evaluate_ardennes(
        datasets_path=_datasets_path,
        config_file=_config_file,
        output_path=_output_path,
        validation_mode=_validation_mode
    )
    # --------------------------------------------------- #
    # correct_mean_height()
