#import tensorflow as tf
import os, sys
#sys.path.append('../')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch as torch
import torch.nn as nn
import pickle as pkl
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold

from torch_CNN_SPN import CNN_SPN_Parts, test_model_SPN_MLP
from CNN_functions_pytorch import load_VAE
from end_to_end_train import load_dataset
from torch_spn import create_torch_spn_parts
import numpy as np


# 0. load model
# 1. evaluate on test
# 2. show example of FP/FP/TP/TN (very certain)


def load_VAE_model( add_info, num_classes, input_shape,data_path_fold,params,path=''):
    """
    Loads  VAE model
    Args:
        add_info (bool): whether additional info channels are used 
        num_classes (int): number of classes in dataset 
        input_shape (tuple): shape of input images (C, H, W)
        data_path_fold (str): path to fold directory
        params (obj): VAE hyperparameters
        path (str): optional root path

    Returns:
        vae_model: loaded VAE pytorch model
    """
    vae_save_path=data_path_fold+'vae_checkpoints_copy/tf_ckpts_last'
    print('TODO _copy')
    # 1. load vae:
    # params,add_info,num_classes,input_shape,checkpoint_path,path='')
    vae_model = load_VAE(
        params = params, 
        add_info = add_info, 
        num_classes = num_classes, 
        input_shape = input_shape, 
        checkpoint_path = vae_save_path,
        path = path)
    return vae_model

def load_model( add_info, num_classes, input_shape,data_path_fold,params,path=''):
    """
    Loads a PyTorch CNN-SPN model, its corresponding VAE embedding model, and SPN structure.

    Args:
        add_info (bool): whether additional info channels are used 
        num_classes (int): number of output classes
        input_shape (tuple): shape of input images (C, H, W)
        data_path_fold (str): path to fold directory
        params (obj): models hyperparameters
        path (str): optional root path

    Returns:
        cnn_spn (nn.Module): Fully loaded PyTorch CNN+SPN model.
    """
    vae_ckpt_path=data_path_fold+'vae_checkpoints/pt_ckpts_last'
    cnn_spn_ckpt_dir = data_path_fold + 'cnn_spn_checkpoints'
    ckpt_file = os.path.join(cnn_spn_ckpt_dir, 'pt_ckpts_last.pt')


    # 1. load vae:
    vae_model = load_VAE(
        params = params, 
        add_info = add_info, 
        num_classes = num_classes, 
        input_shape = input_shape, 
        checkpoint_path = vae_ckpt_path)
    # 2. load spn_structure
    spn_data=pkl.load(open(data_path_fold+ 'spn.pkl', 'rb'))
    spn_clf=spn_data['spn_x']
    spn_input_shape=spn_data['data_shape']
    label_ids=spn_data['label_ids']

    #torch spn
    spn_x_copy, all_spn_x_y, all_spn_x_y_dicts, all_prior, all_spn_x_y_model = \
        create_torch_spn_parts(
            spn_x=spn_clf,
            label_ids=label_ids,
            trainable_leaf=params.fine_tune_leafs,
        )
    # 3. Prepare embedding network
    # Comment out Tensorflow equivalent, replace with pytorch
    decoder = None
    if params.use_VAE:
        cnn_embedding = vae_model.clf_model()
        if params.VAE_fine_tune:
            decoder = vae_model.decoder
    else:
        try:
            cnn_layer_out = vae_model.get_layer('flatten').output
            #cnn_embedding = tf.keras.Model(inputs=vae_model.input, outputs=cnn_layer_out)
            cnn_embedding = nn.Module(inputs=vae_model.input, outputs=cnn_layer_out)
        except:
            raise NotImplementedError("Non-VAE embedding not yet implemented for PyTorch version. Need to get CNN_functions.load_vae() flatten layer first.")


    # 3. load cnn_spn
    cnn_spn = CNN_SPN_Parts(num_classes=num_classes,
                            learning_rate=params.fine_tune_rate,
                            all_spn_x_y_model=all_spn_x_y_model,
                            all_prior=all_prior,
                            cnn=cnn_embedding,
                            get_max=0,
                            gauss_embeds=params.gauss_embeds,
                            use_add_info=params.use_add_info,
                            VAE_fine_tune=params.VAE_fine_tune,
                            decoder=decoder,
                            loss_weights=params.loss_weights,
                            load_pretrain_model=params.load_pretrain_model,
                            clf_mlp=vae_model.classifier)

    # Load checkpoint
    #ckpt_cnn_spn = tf.train.Checkpoint(step=tf.Variable(1), optimizer=cnn_spn.optimizer, net=cnn_spn)
    #manager_cnn_spn = tf.train.CheckpointManager(ckpt_cnn_spn, checkpoint_path_cnn_spn, max_to_keep=1)

    #ckpt_cnn_spn.restore(manager_cnn_spn.latest_checkpoint)
    

    if os.path.isfile(ckpt_file):
        try:
            ckpt_state = torch.load(ckpt_file, map_location="cpu")
            cnn_spn.load_state_dict(ckpt_state)
            print(f"Loaded CNN+SPN checkpoint from {ckpt_file}")
        except Exception as e:
            print(f"WARNING: Failed to load CNN+SPN checkpoint from {ckpt_file}: {e}")
            print("Using randomly initialized CNN+SPN weights.")
    else:
        print(f"WARNING: No CNN+SPN checkpoint found (looked for {ckpt_file}).")
        print("Using randomly initialized CNN+SPN weights.")
    return cnn_spn

def plot_image_grid(images, ax, title,cmap='gray', alpha=1.0):
    """Helper function to plot a 4x4 grid of images."""
    for i in range(16):
        ax[i // 4, i % 4].imshow(images[i], cmap=cmap,alpha=alpha)
        ax[i // 4, i % 4].axis('off')  # Turn off axis for each image
    # Set the header for the grid
    ax[0, 1].set_title(title, fontsize=16, pad=20)

def plot_examples(sorted_data, input_data_name,plot_path,cnn_spn_model):
    for key,value in sorted_data.items():
        vals=value[:16]
        if len(vals)==16:
            # create reconstructions:
            gt, predictions, pred_exp,images=zip(*vals)
            images=np.asarray(list(images))

            test_rec = cnn_spn_model.vae_rec(images)

            # Create a figure with two sets of 4x4 grids (one 8x8 plot with 16 subplots)
            fig, axs = plt.subplots(4, 8, figsize=(12, 6))

            # First grid (left 4x4) for test_rec
            plot_image_grid(test_rec, axs[:, :4], "Reconstructed Images")

            # Second grid (right 4x4) for images
            plot_image_grid(images, axs[:, 4:], "Original Images")

            # Add an overall title
            plt.suptitle(key+' '+input_data_name, fontsize=18)

            # Adjust layout to prevent overlap
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves space for the overall title

            # Display the combined plot
            #plt.show()
            plt.savefig(plot_path+key+'_'+input_data_name+'.png')
            plt.clf()



def analyse_pipeline(dataset_name,params,num_train_eval_runs,data_path_grid,add_info,fold_idxs,model_name,path='../'):
    # 0. load model


    #input_shape=(128,128,3)
    # load data:
    train, test_data, num_classes = load_dataset(dataset_name, binary=False,
                                                                     load_net=params.load_pretrain_model,
                                                                     machine=params.machine,path=path)
    statistics_names=['Accuracy','AUC','Rec:MSE']
    result_clf=[]
    result_rec=[]

    for f_i in fold_idxs:
        data_path_fold = data_path_grid + 'fold_' + str(f_i) + '/'

        kf = KFold(n_splits=num_train_eval_runs, random_state=1, shuffle=True)

        for i,(train_index, val_index) in enumerate(kf.split(train[0])):
            if i ==f_i:

                random.seed(0)
                random.shuffle(train_index)
                random.shuffle(val_index)
                train_x, val_x = train[0][train_index], train[0][val_index]
                train_y, val_y = train[1][train_index], train[1][val_index]
                train_data=(train_x,train_y)

        for input_data , input_data_name in zip([test_data,train_data],['Test','Train']):
            print(input_data_name)

            input_shape = train_data[0].shape[1:]
            cnn_spn_model=load_model( add_info, num_classes, input_shape,data_path_fold,params,path='../')

            # 1. evaluate on test

            results_MLP,results_SPN,losses = test_model_SPN_MLP(cnn_spn_model, input_data, num_classes=num_classes,
                                                            batch_size=params.batch_size,
                                                            training=False, add_info=add_info)

            for result, clf_name in zip([results_MLP,results_SPN],['MLP','SPN']):
                print(result,flush=True)
                result_clf.append({
                    'split':input_data_name,
                    "dataset": dataset_name,
                    "clf": clf_name,
                    "fold_idx": f_i,
                    "model_name": model_name,
                    'Accuracy':result[0],
                    'Entropy':result[1],
                    'Balanced Accuracy':result[2],
                    'Precision':result[3],
                    'Recall':result[4],
                    'F1-Score':result[5],
                    'AUC':result[6]
                })
            result_rec.append({
                'split': input_data_name,
                "dataset": dataset_name,
                "fold_idx": f_i,
                "model_name": model_name,
                'MSE':losses[1],
                'MAE':losses[4],
                'KLD':losses[3],
            })

    return result_clf,result_rec



