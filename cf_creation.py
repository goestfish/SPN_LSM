import argparse
import pickle as pkl
from counterfactuals.find_many_counterfactuals import multiple_vanilla_counterfactuals

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process a file path.")

    # Add the --filepath argument
    parser.add_argument(
        "--filepath",
        type=str,
        default='test_model',
        # required=True,
        help="Path to the input file."
    )
    work_from=0

    num_train_eval_runs = 3
    grid_idx = 0
    add_info = 1

    # Parse the command-line arguments
    args = parser.parse_args()
    dataset_name='chexpert'

    save_folder ='cnn_spn_models/'

    # Access the filepath
    model_name = args.filepath


    data_path_grid = save_folder + model_name + '/grid' + str(grid_idx) + '/'
    params = pkl.load(open(data_path_grid + 'grid_params.pkl', 'rb'))

    for fold_idx in range(num_train_eval_runs):
        multiple_vanilla_counterfactuals(dataset_name, params, data_path_grid, add_info, model_name,
                                         num_imgs=1000,
                                         path='',
                                         learning_rate=0.001,
                                         num_steps=1001,
                                         fold_idx=fold_idx)