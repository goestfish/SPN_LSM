import numpy
import numpy as np
from CNN_functions_pytorch import train_embedding_cnn,get_layer_embeddings,load_CNN,train_embedding_VAE,load_VAE_and_eval
from SPN_functions import create_SPN
import pickle as pkl
from torch_CNN_SPN import  CNN_SPN_Parts, test_model_no_mpe, train_model_parts
from spn.algorithms.Statistics import get_structure_stats
from tf2_spn import create_tf_spn_parts
import os
import gc
import shutil
import torch

from utils import data_to_batch


def load_dataset(dataset_name,binary,load_net,machine,path='',grid_params=None):

    data_path = path+'chexpert.pkl'
    data = pkl.load(open(data_path, 'rb'))
    #[X, int_data, float_data]=val
    train_data=data['train']
    #'train', 'validation', 'test'
    val_data=data['validation']
    test_data=data['test']

    #normalize float data
    max_floats=np.max(train_data[2],axis=0)
    train_data=[train_data[0],np.concatenate([train_data[1],train_data[2]/max_floats],axis=1)]
    val_data = [val_data[0], np.concatenate([val_data[1], val_data[2] / max_floats], axis=1)]
    test_data = [test_data[0], np.concatenate([test_data[1], test_data[2] / max_floats], axis=1)]
    print(val_data[0].shape,val_data[1].shape)
    if not load_net:
        train_data[0]=np.expand_dims(train_data[0]/255.0,axis=-1)
        test_data[0]=np.expand_dims(test_data[0]/255.0,axis=-1)
        val_data[0] =np.expand_dims( val_data[0] / 255.0,axis=-1)
    else:
        train_data[0]=np.repeat(np.expand_dims(train_data[0],axis=-1), 3, axis=3)
        test_data[0]=np.repeat(np.expand_dims(test_data[0],axis=-1), 3, axis=3)
        val_data[0] = np.repeat(np.expand_dims(val_data[0], axis=-1), 3, axis=3)
    print('load chex')

    train_new = (np.concatenate([train_data[0], val_data[0]], axis=0), np.concatenate([train_data[1], val_data[1]], axis=0))
    return train_new,test_data,2



def SPN_structure_train(grid_params,model, layer_names, train_dataset, test_dataset,get_max,num_test_instances,
                        num_classes,debugging,add_info):
    embedding_arr, all_labels ,other_info= get_layer_embeddings(grid_params,model, layer_names, [train_dataset, test_dataset],
                                                     get_max=get_max,add_info=add_info)  # valloader

    # normalize embedding:

    [train_embedding, test_embedding] = embedding_arr
    if grid_params.use_add_info:
        print(other_info[0].shape)
        train_embedding=np.concatenate([train_embedding,other_info[0]],axis=-1)
        test_embedding = np.concatenate([test_embedding, other_info[1]], axis=-1)

    # 3. create spn structure +evaluate SPN structure

    train_data_spn = np.concatenate([np.expand_dims(all_labels[0], axis=-1), train_embedding], axis=-1)
    out_labels = np.full((num_test_instances, 1), np.nan)
    test_data_spn = np.concatenate([out_labels, test_embedding], axis=1)

    if num_classes==1:
        regression=grid_params.regression
        spn_clf, label_ids=[],[]
        for pseudo_cls in range(regression):

            sub_spn_clf, sub_label_ids = create_SPN(train_data_spn, test_data_spn, all_labels[1],
                                            input_dim=train_embedding.shape[1])
            spn_clf.append(sub_spn_clf)
            label_ids.append(sub_label_ids)
    else:
        spn_clf,label_ids = create_SPN(grid_params,train_data_spn, test_data_spn, all_labels[1], input_dim=train_embedding.shape[1],debugging=debugging)
    print('STRUCTURE STATS', flush=True)
    print(get_structure_stats(spn_clf))
    return spn_clf,label_ids




def train_cnn_spn(grid_params,train_data,test_data,val_data,num_classes,debugging=False,split=0,fold_path=''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    get_max=False
    load_data=False
    max_iterations=1#10
    model_path=fold_path+'vae_checkpoints/'#'models/'+model_name
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    checkpoint_path = model_path + 'pt_ckpts_last'

    model_path=fold_path+'vae_checkpoints_tmp/'#'models/'+model_name
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    checkpoint_path_tmp = model_path + 'pt_ckpts_last'
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(checkpoint_path_tmp, exist_ok=True)

    checkpoint_path_cnn_spn=fold_path+'cnn_spn_checkpoints'
    if not os.path.exists(checkpoint_path_cnn_spn):
        os.mkdir(checkpoint_path_cnn_spn)
    checkpoint_path_cnn_spn = checkpoint_path_cnn_spn + '/pt_ckpts_last'



    batch_size=grid_params.batch_size
    fine_tune_rate=grid_params.fine_tune_rate
    last_num_filters = grid_params.latent_dim
    layer_names = ['flatten']

    # 1. train embedding CNN + evaluate performance of CNN
    # save CNN weights
    add_info=grid_params.use_add_info
    print('add info: spn training',add_info)

    input_shape = train_data[0].shape[1:] #(112,112,3)#
    print(input_shape)


    if grid_params.use_VAE:
        model,manager,ckpt,_,all_debugging_stuff=train_embedding_VAE(grid_params,train_data,test_data,val_data,input_shape,
                                               num_classes,checkpoint_path,load_data,split,add_info,checkpoint_path_tmp)
        # save debugging stuff:
        if grid_params.VAE_debug:
            print('save debugging file')
            pkl.dump(all_debugging_stuff,open(fold_path+'_debugg.pkl','wb'))
    else:
        model,manager,ckpt=train_embedding_cnn(grid_params,val_data,input_shape,
                                               num_classes,checkpoint_path,load_data,split,train_data,add_info)

    # save cnn + mlp in another folder:
    ##################################
    model_path_copy=fold_path+'vae_checkpoints_copy/'#'models/'+model_name
    if not os.path.exists(model_path_copy):
        os.mkdir(model_path_copy)
    model_path_copy+= '/pt_ckpts_last'
    if not os.path.exists(model_path_copy):
        os.mkdir(model_path_copy)
    # Copy each file from dir1 to dir2
    for filename in os.listdir(checkpoint_path+'/'):
        shutil.copy2(checkpoint_path+'/'+filename, model_path_copy+'/'+filename)

    gc.collect()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    ##########################################################
    # 5. create Cnn_spn
    #acc_diff=1
    it_counter=0
    all_accs=[]


    curr_acc=0
    MLP_evals=[]
    all_vae_debug_stuff=[]

    train_dataset,test_dataset,val_dataset=data_to_batch(train_data,test_data,val_data,add_info,batch_size)


    while it_counter<max_iterations:
        # load saved model:
        if grid_params.use_VAE:
            model,ckpt,manager,MLP_eval=load_VAE_and_eval(grid_params, input_shape, num_classes,
                                                 debugging,ckpt,manager,checkpoint_path,
                                                 train_dataset,test_dataset,val_dataset,add_info)
        else:
            model,ckpt,manager,MLP_eval=load_CNN(grid_params, input_shape, num_classes,
                                                 debugging,ckpt,manager,checkpoint_path,
                                                 train_data,test_data,val_data,add_info)

        print('MLP eval',MLP_eval)
        MLP_evals.append(MLP_eval)

        spn_clf,label_ids = SPN_structure_train(grid_params,model, layer_names, train_dataset, test_dataset, get_max,
                                      num_test_instances=test_data[0].shape[0],num_classes=num_classes,debugging=debugging,add_info=add_info)

        decoder=None
        if grid_params.use_VAE:
            cnn_embedding = model.clf_model()
            if grid_params.VAE_fine_tune:
                decoder=model.decoder
        else:
            cnn_embedding = model.get_feature_extractor()

        if grid_params.use_add_info and add_info:
            spn_input_shape=(last_num_filters + train_data[1].shape[1],)
        else:
            spn_input_shape=(last_num_filters + 1,)

        spn_x_copy,all_spn_x_y,all_spn_x_y_dicts,all_prior,all_spn_x_y_model= create_tf_spn_parts(spn_x=spn_clf,
                                                                                                  data_shape=spn_input_shape,
                                                                                                  label_ids=label_ids
                                                                                                  ,trainable_leaf=grid_params.fine_tune_leafs)
        cnn_spn = CNN_SPN_Parts(num_classes=num_classes,
                                learning_rate=fine_tune_rate,
                                all_spn_x_y_model=all_spn_x_y_model,
                                all_prior=all_prior,
                                cnn=cnn_embedding,
                                get_max=get_max,
                                gauss_embeds=grid_params.gauss_embeds,
                                use_add_info=grid_params.use_add_info,
                                VAE_fine_tune=grid_params.VAE_fine_tune,
                                decoder=decoder,
                                loss_weights=grid_params.loss_weights,
                                load_pretrain_model=grid_params.load_pretrain_model,
                                use_GAN=grid_params.GAN,
                                filter_size=grid_params.filter_size,
                                num_layer=grid_params.num_layer,
                                batchnorm_integration=grid_params.batchnorm_integration,
                                shortcut=grid_params.shortcut,
                                activation=grid_params.activation,
                                num_filter=grid_params.num_filter_encoder,
                                strides=grid_params.strides_encoder,
                                end_dim=100,
                                dropout=grid_params.dropout,clf_mlp=model.classifier
                                )
        cnn_spn = cnn_spn.to(device)


        #########################################################################################################
        spn_save_data={'spn_x' : spn_clf,
                       'data_shape' : spn_input_shape,
                       'label_ids' : label_ids}
        pkl.dump(spn_save_data, open(fold_path+ 'spn.pkl', 'wb'))
        # save SPN weights
        # checkpoint file:



        #########################################################################################################


        #print('FIRST SPN TEST',flush=True)
        print('BEFORE CNN+SPN TRAIN')
        eval_before_train=test_model_no_mpe(cnn_spn,  [train_data,val_data,test_data],num_classes,grid_params.batch_size,add_info=add_info)
        if eval_before_train[1][2]> curr_acc:
            curr_acc=eval_before_train[1][2]


        # 6 train the method and fine tune both CNN + SPN
        print('TRAIN CNN+SPN',flush=True)
        eval_after_train,vae_debugg_stuff=train_model_parts(grid_params,cnn_spn, train_data, val_data,test_data,
                          num_iterations=grid_params.fine_tune_its,
                          ckpt=[],manager=[],val_entropy=eval_before_train[1][1],val_acc=curr_acc,add_info=add_info)
        all_vae_debug_stuff.append(vae_debugg_stuff)
        # save this:
        # save debugging stuff:
        if grid_params.VAE_fine_tune:
            print('save debugging file')
            pkl.dump(all_vae_debug_stuff,open(fold_path+'_SPN_debug.pkl','wb'))


        if not len(eval_after_train):
            eval_after_train=eval_before_train



        if eval_after_train[1][0]> curr_acc:
            curr_acc=eval_after_train[1][0]
            #ckpt_cnn_spn.step.assign_add(1)
            #manager_cnn_spn.save()



        eval_diff=eval_after_train[1][0]-eval_before_train[1][0]

        all_accs.extend([eval_before_train,eval_after_train])

        print('ACC increase:',eval_after_train[1][0],eval_before_train[1][0],eval_diff,flush=True)#eval_after_train,eval_before_train


        del cnn_spn
        del spn_clf
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        it_counter+=1



    return all_accs,MLP_evals


