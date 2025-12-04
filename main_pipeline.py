import spflow_patches #bowei

import random
import argparse

import numpy as np
from sklearn.model_selection import ParameterGrid
from munch import DefaultMunch
import pickle as pkl
import time
import os
from sklearn.model_selection import KFold

from end_to_end_train import train_cnn_spn, load_dataset


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--machine', default=['laptop'], type=str, nargs='+')
    #CNN prams
    argparser.add_argument('--dense_lay_num', default=[128], type=int, nargs='+')#15
    argparser.add_argument('--le_warmup', default=[1], type=int, nargs='+')
    argparser.add_argument('--gauss_std', default=[0.2], type=float, nargs='+')#0.00005

    ################################################################################################################################
    # CNN /VAE params:
    argparser.add_argument('--save_path', default=["test_model"], type=str, nargs='+')
    argparser.add_argument('--epochs', default=[100], type=int, nargs='+')  # 15#1000
    argparser.add_argument('--latent_dim', default=[64], type=int, nargs='+')#16

    argparser.add_argument('--load_pretrain_model', default=[0], type=int, nargs='+')
    argparser.add_argument('--dropout', default=[0.0], type=float, nargs='+')
    argparser.add_argument('--learning_rate', default=[0.0005], type=float, nargs='+')#0.00005
    argparser.add_argument('--batch_size', default=[100], type=int, nargs='+')
    argparser.add_argument('--num_layer', default=[4], type=int, nargs='+')#6
    argparser.add_argument('--filter_size', default=[3], type=int, nargs='+')
    argparser.add_argument('--end_dim_enc', default=[128], type=int, nargs='+')#15

    argparser.add_argument('--use_VAE', default=[1], type=int, nargs='+')
    argparser.add_argument('--use_KLD_anneal', default=[0], type=int, nargs='+')
    #VAE params:
    argparser.add_argument('--num_filter_encoder', default=[[16,32, 62, 124, 124, 124, 124]], type=int, nargs='+')
    argparser.add_argument('--strides_encoder', default=[[2, 1, 2, 1, 2, 1, 2]], type=int, nargs='+')
    argparser.add_argument('--num_filter_decoder', default=[[1,62,62,124,124,124,248]], type=int, nargs='+')
    argparser.add_argument('--strides_decoder', default=[[2,1,2,1,2,1,2]], type=int, nargs='+')

    argparser.add_argument('--semi_supervised', default=[1], type=int, nargs='+')
    argparser.add_argument('--batchnorm_integration', default=[0], type=int, nargs='+')
    argparser.add_argument('--shortcut', default=[1], type=int, nargs='+')
    argparser.add_argument('--activation', default=['relu'], type=str, nargs='+',help='["relu","tanh"]')
    argparser.add_argument('--VAE_debug', default=[1], type=int, nargs='+')
    #(rec_loss * self.loss_weights[0]) + (kl_loss * self.loss_weights[1]) + (clf_loss *self.loss_weights[2])
    argparser.add_argument('--loss_weights', default=[[10.0, 0.001, 5.0]], type=float, nargs='+')#[10.0, 0.0005, 5.0]
    argparser.add_argument('--VAE_fine_tune', default=[1], type=int, nargs='+')
    argparser.add_argument('--GAN', default=[0], type=int, nargs='+')
    # 0 = no fine tuning, 1= fine-tune in one loss 2= fine tune in two separate steps


    #################################################################################################################
    #SPN params:
    argparser.add_argument('--fine_tune_its', default=[0], type=int, nargs='+')
    argparser.add_argument('--fine_tune_rate',default=[0.0005], type=float, nargs='+')#[0.001,0.0001,0.0001,]
    argparser.add_argument('--regression', default=[10], type=int, nargs='+')
    argparser.add_argument('--min_instances_slice_percentage', default=[0.2], type=float, nargs='+')
    argparser.add_argument('--min_features_slice', default=[1], type=int, nargs='+')
    argparser.add_argument('--n_clusters', default=[2], type=int, nargs='+')

    argparser.add_argument('--row_split', default=["kmeans"], type=str, nargs='+',help='["rdc","kmeans","tsne","gmm"]')
    argparser.add_argument('--col_split', default=["rdc"], type=str, nargs='+', help='["rdc","poisson"]')
    argparser.add_argument('--fine_tune_leafs', default=[1], type=int, nargs='+')
    argparser.add_argument('--gauss_embeds', default=[0.05], type=float, nargs='+')

    argparser.add_argument('--use_add_info', default=[1], type=int, nargs='+')

    argparser.add_argument('--separate_view', default=[0], type=int, nargs='+')

    argparser.add_argument('--dataset_name', default=['chexpert'], type=str, nargs='+', help='')#'CXR_process_org_distr','chexpert'

    arguments = argparser.parse_args()
    configs=dict()
    print('####Used params:')
    for k,v in arguments.__dict__.items():
        if k=='loss_weights':
            v=[v]
        configs[k]=v
        print(k,v)
    grid = ParameterGrid(configs)
    grid=[DefaultMunch.fromDict(entry) for entry in  grid]
    return grid


if __name__ == '__main__':
    grid_start=0

    debug=True

    param_grid=parse_args()
    num_train_eval_runs = 3

    result_curves=[]
    all_MLP_evals=[]
    param_grid=param_grid[grid_start:grid_start+5]
    print('Grid len',len(param_grid))

    save_folder='cnn_spn_models/'
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    for grid_idx,grid_params in enumerate(param_grid):
        res_file_path = grid_params.save_path+'.pkl'
        model_name = grid_params.save_path
        save_folder_grid=save_folder + model_name+'/grid'+str(grid_idx)+'/'
        if not os.path.isdir(save_folder+model_name):
            os.mkdir(save_folder+model_name)
            os.mkdir(save_folder_grid)
        else:
            if not os.path.isdir(save_folder_grid):
                os.mkdir(save_folder_grid)
        # save grid_data:
        pkl.dump(grid_params,open(save_folder_grid+'grid_params.pkl','wb'))
        dataset_name=grid_params.dataset_name
        print('dataset name:', dataset_name)
        train, test, num_classes = load_dataset(dataset_name, binary=False,
                                                                         load_net=grid_params.load_pretrain_model,
                                                                         machine=grid_params.machine,grid_params=grid_params)
        kf = KFold(n_splits=3, random_state=1,shuffle=True)
        for i, (train_index, val_index) in enumerate(kf.split(train[0])):
            fold_path=save_folder_grid+'fold_'+str(i)+'/'
            if not os.path.isdir(fold_path):
                os.mkdir(fold_path)

            random.seed(0)
            random.shuffle(train_index)
            random.shuffle(val_index)
            train_x,val_x=train[0][train_index],train[0][val_index]
            train_y, val_y =train[1][train_index], train[1][val_index]
            time_start=time.time()
            res_curves,MLP_eval=train_cnn_spn(grid_params,(train_x, train_y), test,(val_x,val_y),
                                              num_classes,debug,split=i,fold_path=fold_path)

            print('EVAL TIME OF ONE SPLIT:',(time.time()-time_start)//60)

            result_curves.append(res_curves)
            all_MLP_evals.append(MLP_eval)

            pkl.dump([result_curves,param_grid,all_MLP_evals],open(res_file_path,'wb'))