import pickle as pkl
import argparse
from counterfactuals.analyse_model import analyse_pipeline
from end_to_end_train import load_dataset
from evaluation.counterfactual_evaluation import eval_counterfactuals, df_to_latex_summary, \
    performance_to_latex_summary, cf_performance_to_latex_summary, z_eval_cf_performance_to_latex_summary, plot_cfs
import os
import pandas as pd
# Set the cache directory for TensorFlow Hub
os.environ["TFHUB_CACHE_DIR"] = "tfhub_modules"


def get_performance_metrics(model_names, fold_idxs, num_train_eval_runs=3, grid_idx=0, add_info=1,path=''):
    all_results_clf = []
    all_results_rec = []
    all_params = {}

    for model_name in model_names:
        print(model_name)

        data_path_grid = save_folder + model_name + '/grid' + str(grid_idx) + '/'
        params_path = os.path.join(data_path_grid, 'grid_params.pkl')

        if not os.path.exists(params_path):
            print(f"Parameter file not found for {model_name}, skipping...")
            continue

        params = pkl.load(open(params_path, 'rb'))
        all_params[model_name] = params

        # Check if results already exist
        rec_path = os.path.join(data_path_grid, "rec.csv")
        clf_path = os.path.join(data_path_grid, "clf.csv")

        if os.path.exists(rec_path) and os.path.exists(clf_path):
            print(f"Loading existing results for {model_name}")
            result_rec = pd.read_csv(rec_path).to_dict('records')
            result_clf = pd.read_csv(clf_path).to_dict('records')
           # print(pd.read_csv(rec_path)[:5])
           # print(result_rec[:5])

        else:
            print(f"Running analysis for {model_name}")
            result_clf, result_rec = analyse_pipeline(dataset_name, params, num_train_eval_runs, data_path_grid,
                                                      add_info, fold_idxs, model_name,path=path)

            # Save results
            pd.DataFrame(result_rec).to_csv(rec_path, index=False)
            pd.DataFrame(result_clf).to_csv(clf_path, index=False)
            print('Results saved')

        all_results_rec.extend(result_rec)
        all_results_clf.extend(result_clf)

    # Print to LaTeX
    performance_to_latex_summary(all_results_clf, all_results_rec, all_params)





def get_cf_metrics(model_names,possible_values,fold_idxs):
    all_results= []
    all_results_z_eval=[]
    all_params = {}

    for model_name in model_names:
        print(model_name)
        #num_train_eval_runs=3
        grid_idx=0

        data_path_grid=save_folder + model_name+'/grid'+str(grid_idx)+'/'
        params = pkl.load(open(data_path_grid + 'grid_params.pkl', 'rb'))


        train, test, num_classes = load_dataset(dataset_name, binary=False,
                                                                         load_net=params.load_pretrain_model,
                                                                         machine=params.machine,grid_params=params)

        # Check if results already exist
        cf_path = os.path.join(data_path_grid, "cf_new.csv")#1
        z_eval_path = os.path.join(data_path_grid, "z_eval_new.csv")#1
        #'''
        if os.path.exists(cf_path):
            with open(cf_path, "r") as f:
                if f.read(1).replace('\n','').replace('\t','')== "":
                    print("File is empty")
                    print(f"Running analysis for {model_name}")
                    cf_result,results_z_eval  = eval_counterfactuals(dataset_name, data_path_grid, model_name, fold_idxs=fold_idxs,
                                                     possible_values=possible_values,test=test)
                    pd.DataFrame(cf_result).to_csv(cf_path, index=False)
                    pd.DataFrame(results_z_eval).to_csv(z_eval_path, index=False)
                    print('Results saved')
                else:
                    print("File is not empty")
                    print(f"Loading existing results for {model_name}")
                    cf_result = pd.read_csv(cf_path).to_dict('records')
                    results_z_eval= pd.read_csv(z_eval_path).to_dict('records')
        else:
        #'''
            print(f"Running analysis for {model_name}")
            cf_result,results_z_eval = eval_counterfactuals(dataset_name, data_path_grid, model_name, fold_idxs=fold_idxs,possible_values=possible_values,test=test)
            print(cf_result)
            pd.DataFrame(cf_result).to_csv(cf_path, index=False)
            pd.DataFrame(results_z_eval).to_csv(z_eval_path, index=False)
            print('Results saved')


        all_results.extend(cf_result)
        all_results_z_eval.extend(results_z_eval)
        all_params[model_name]=params

    cf_performance_to_latex_summary(all_results, all_params)
    z_eval_cf_performance_to_latex_summary(all_results_z_eval, all_params)

def plot_generated_cfs(model_names,possible_values, fold_idxs):

    for model_name in model_names:
        print(model_name)
        #num_train_eval_runs=3
        grid_idx=0

        data_path_grid=save_folder + model_name+'/grid'+str(grid_idx)+'/'
        params = pkl.load(open(data_path_grid + 'grid_params.pkl', 'rb'))
        for fold_idx in fold_idxs:


            plot_cfs(dataset_name,
                        data_path_grid,
                        model_name,
                        fold_idxs=[fold_idx],
                        possible_values=possible_values)


if __name__ == '__main__':


    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process a file path.")

    # Add the --filepath argument
    parser.add_argument(
        "--filepath", 
        type=str,
        default='test_model',
        #required=True,
        help="Path to the input file."
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the filepath
    model_name= args.filepath



    dataset_name='chexpert'
    fold_idxs = list(range(3))

    save_folder ='cnn_spn_models/'

    model_names=['full_run_100e_ft50']#,'cnn_spn_model_test'#['cnn_spn_model_test']#['full_run_100e_ft50']#



    get_performance_metrics(model_names,fold_idxs,path='')
    possible_values=[0,1]
    get_cf_metrics(model_names,possible_values,fold_idxs)
    plot_generated_cfs(model_names, possible_values, fold_idxs)





