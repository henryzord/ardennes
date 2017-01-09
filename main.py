import json

# noinspection PyUnresolvedReferences
from evaluate import evaluate_ardennes, evaluate_j48, crunch_graphical_model, \
    grid_optimizer, crunch_parametrization, crunch_result_file, do_train

if __name__ == '__main__':
    _config_file = json.load(open('config.json', 'r'))
    _datasets_path = 'datasets/numerical'

    _folds_path = 'datasets/folds'
    _output_path = 'metadata'
    _validation_mode = 'cross-validation'
    _intermediary_sets = 'intermediary'

    evaluate_ardennes(
        datasets_path=_datasets_path,
        config_file=_config_file,
        output_path=_output_path,
        validation_mode=_validation_mode
    )
    # evaluate_j48(_datasets_path, _intermediary_sets)
    # # --------------------------------------------------- #
    # crunch_graphical_model(
    #     '/home/henryzord/Projects/ardennes/metadata/liver-disorders/liver-disorders_pgm_fold_000_run_000.csv',
    #     _datasets_path
    # )
    # # --------------------------------------------------- #
    # grid_optimizer(_config_file, _datasets_path, output_path='/home/henry/Desktop/parametrizations')
    # # --------------------------------------------------- #
    # crunch_parametrization('parametrization_hayes-roth-full.csv')
    # # --------------------------------------------------- #
    # _results_file = json.load(
    #     open('/home/henry/Desktop/results.json', 'r')
    # )
    # crunch_result_file(_results_file, output_file='results.csv')
    # # --------------------------------------------------- #
    # _evaluation_mode = 'holdout'
    # do_train(config_file=_config_file, n_run=0, evaluation_mode=_evaluation_mode)
    # --------------------------------------------------- #
