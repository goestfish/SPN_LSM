from counterfactuals.analyse_model import load_model, load_VAE_model
import os
import sys
sys.path.append('../')

import numpy as np
from end_to_end_train import load_dataset
import pickle as pkl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
#import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
from concurrent.futures import ProcessPoolExecutor
import time
from multiprocessing import Pool, set_start_method


def nan_to_num(tensor,nan,posinf):
    """
    Replace NaNs and positive infinities in a torch tensor

    Args:
        tensor (torch.Tensor or np.ndarray): input
        nan (float): value to replace NaNs with
        posinf (float): value to replace positive infinities with

    Returns:
        tensor: cleaned Torch.tensor
    """
    #is_pos_inf = tf.math.is_inf(tensor) & (tensor > 0)
    #tensor= tf.where(is_pos_inf, posinf, tensor)
    #return tf.where(tf.math.is_nan(tensor), nan, tensor)
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)
    # first dealing with positive infinity values
    is_pos_inf = torch.isinf(tensor) & (tensor > 0)
    tensor = torch.where(is_pos_inf, torch.tensor(posinf, dtype = tensor.dtype, device = tensor.device), tensor)
    # and now dealing with nan's
    tensor = torch.where(torch.isnan(tensor), torch.tensor(nan, dtype = tensor.dtype, device = tensor.device), tensor)
    return tensor



def get_counterfactual_infos(img_idx,cnn_spn_model,X,additional_info,opposite_class,y,model_type,opt_weights=[10,0.005],learning_rate = 0.01,num_steps = 150):
    """ Get the latent variable z for the input. Generate counterfactual latent vectors z' by optimizing z' so that the classifier's prediction
    moves to the opposite class while keeping z' close to the original z and latent likelihood similar.

    Args:
        img_idx (int): index of image in dataset. Optional for logging
        cnn_spn_model: Pytorch wrapper providing:
                        - embedding(X) --> (z_embed, ... , ...)
                        - spn_clf(embedding) --> (pred_logits, p_z)
                        - reconstruct(z_tensor) --> reconstructed images tensor
        X (np.ndarray or torch.Tensor): Input batch 
        additional_info (np.ndarray or torch.Tensor): additional info per sample 
        opposite_class (int): new target class to push to (0,1)
        y(np.ndarray or list): original labels (batch or single)
        model_type (str): 'MLP' or 'SPN' - determines loss formulation 
        opt_weights (list): weights [beta, gamma] for distance and plausibility metrics 
        learning_rate (float): optimizer learning rate for optimizing z'
        num_steps (int): number of gradient steps on z'
    Returns:
        reconstruction_np, rec_z_np, z_prime, np, title_info, distance_np, arg_map_np, loss_val, log_pred_val, p_z_np, label_switch_step, pred_np 
    """
    # get device from model if possible
    device = None
    try:
        # prefer model parameters
        params = next(cnn_spn_model.parameters())
        device = params.device
    except StopIteration:
        device = torch.device('cpu')
    # make sure we're working with tensors that are optimized for our device
    if not torch.is_tensor(X):
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
    else:
        X_t = X.to(device)
    if not torch.is_tensor(additional_info):
        add_info_t = torch.tensor(additional_info, dtype = torch.float32, device = device)
    else:
        add_info_t = additional_info.to(device)

    # New: Get latent embedding z for the input (expecting a torch tensor?)
    # And the embedding should return (z_embed, ..., ...)
    #print('image',img_idx)
    #[z_embed, _, _] = cnn_spn_model.embedding(X, training=False)
    # z = vae_encoder(input_x)

    # Define z_prime as a trainable variable initialized from z
    #y_opposite=tf.Variable(np.asarray([opposite_class]*z_embed.shape[0]), trainable=True, dtype=tf.dtypes.int32)
    #z = tf.Variable(z_embed, trainable=False, dtype=tf.dtypes.float32)
    #z_prime = tf.Variable(z_embed, trainable=True, dtype=tf.dtypes.float32)
    #add_info = tf.Variable(additional_info, trainable=False, dtype=tf.dtypes.float32)
    #print('shapes', z_prime.shape, additional_info.shape,'le:',learning_rate)

    with torch.no_grad():
        z_embed_tuple = cnn_spn_model.embedding(X_t) #expecting (z, ... , ...)
        # account for if the embedding returns a tuple or list
        if isinstance(z_embed_tuple, tuple) or isinstance(z_embed_tuple, list):
            z_embed = z_embed_tuple[0]
        else:
            z_embed = z_embed_tuple
    # New: detach original z
    z = z_embed.detach().to(device) # new shape expected (batch, latent_dim)
    # New: crate z_prime as a trainable parameter initialized to z
    z_prime = z.clone().detach().requires_grad_(True)
    # New: Create y_opposite tensor for classification loss (if we need it)
    batch_size = z.shape[0]
    y_opposite = torch.full((batch_size,), opposite_class, dtype = torch.long, device = device)


    # Define the optimizer
    #optimizer = tf.keras.optimizers.SGD(learning_rate)  # Adam(learning_rate)
    #myloss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = torch.optim.SGD([z_prime], lr=learning_rate)
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    # Loss function to push z' to the opposite class
    
    # pre-initializing some variables for later
    title_info = ''
    arg_max = None
    loss_val = None
    log_pred_val = None
    distance_val = None
    p_z_val = None
    label_switch_step = -1
    switch = False
    pred = None
    
    # Precompute embedding_org related values
    embedding_org = torch.cat([z, add_info_t], dim=-1)
    with torch.no_grad():
        pred_org, p_z_org = cnn_spn_model.spn_clf(embedding_org) # pred_org logits, p_z_org maybe log probs
    
    #optimization loop
    for step in range(num_steps):
        optimizer.zero_grad()
         
         # create embedding from z_prime + additional info
        embedding = torch.cat([z_prime, add_info_t], dim=-1)

         # get predictions and p(z) from SPN or MLP (expecting logits)
        pred_logits, p_z = cnn_spn_model.spn_clf(embedding) # pred_logits: (batch_num_classes)

         # compute distance per sample (mean squared on latent dims)
        distance = torch.mean((z-z_prime) ** 2, dim=-1) # (batch, )
        log_pred = None
        if model_type == 'MLP':
            # CrossEntropy expects (batch, classes) logits and (batch,) targets
            test = ce_loss(pred_logits, y_opposite) # per-sample loss
            # total per sample loss = CE + beta * distance
            per_sample_loss = test + (distance * opt_weights[0])
            loss = torch.mean(per_sample_loss)
            # for logging purposes -- set log_pred as negative cross_entropy (higher is better)
            log_pred = -test.detach()
        elif model_type == 'SPN':
            # assuming pred_logits are class-scores (bigger --> more probable); use opposite_class column
            # test is a per-sample score = pred_logits[:, opposite_class]
            test = pred_logits[:, opposite_class]
            # plausibility difference between p_z and p_z_org
            # ensure they're tensors w/ same shape
            # compute per-sample sum of abs difference across appropriate dim
            try:
                plaus_diff = torch.sum(torch.abs(p_z - p_z_org), dim=-1)
            except Exception:
                # fallback if p_z are scalars
                plaus_diff = torch.abs(p_z - p_z_org).reshape(-1)
            per_sample_loss = (-test) + (distance * opt_weights[0]) + (plaus_diff * opt_weights[1])
            loss = torch.mean(per_sample_loss)
            log_pred = -test.detach()

        else:
            raise ValueError("model_type must be either 'MLP' or 'SPN'")
    # Backprop on z_prime
    loss.backward()
    optimizer.step()
    # Evaluate switching condition
    with torch.no_grad():
        argmax_vals = torch.argmax(pred_logits, dim=1).cpu().numpy()
        argmax_mean = np.mean(argmax_vals)

        # if opposite_class = 1 & mean >= 0.5 OR opposite_class = 0 & mean <= 0.5
        if not switch:
            # could combine these two conditions into the same if statement but this is more readable?
            if opposite_class and argmax_mean >= 0.5: 
                switch = True
                label_switch_step = step
            elif (not opposite_class) and argmax_mean <= 0.5:
                switch = True
                label_switch_step = step
            # else switch stays False
    # if last step, gather diagnostics
    if step == num_steps - 1:
        with torch.no_grad():
            argmax = argmax_vals
            loss_val = loss.detach().cpu().item()
            # log_pred - used nefgative per-sample test; take mean
            if isinstance(log_pred, torch.Tensor):
                try:
                    log_pred_val = log_pred.mean().cpu().numpy()
                except Exception:
                    log_pred_val = float(log_pred)
            else:
                log_pred_val = float(log_pred)
            distance_val = distance.detach().cpu().numpy()
            p_z_val = p_z.detach().cpu().numpy() if torch.is_tensor(p_z) else np.array(p_z)
            arg_max_mean = float(argmax_mean)
            distance_mean = float(np.mean(distance_val))
            p_z_mean = float(np.mean(p_z_val)) if np.size(p_z_val) else float(p_z_val)
            title_info = f"New y:{arg_max_mean:.0f}; orginal y:{int(y[0])}; MSE{distance_mean:.2f}; log(p(z)):{p_z_mean:.2f}"            

    # After optimization, reconstruct images from z_prime and z
    with torch.no_grad():
        # make sure z_prime is on the device and detach
        z_prime_out = z_prime.detach()
        z_out = z.detach()
        reconstructions = cnn_spn_model.reconstruct(z_prime_out) # expecting a tensor here
        rec_z = cnn_spn_model.reconstruct(z_out)
    # convert outpuits to numpy for later stuff that expects numpy
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    reconstructions_np = to_numpy(reconstructions)
    rec_z_np = to_numpy(rec_z)
    z_prime_np = to_numpy(z_prime_out)
    z_np = to_numpy(z_out)
    pred_np = to_numpy(pred_logits.detach()) if 'pred_logits' in locals() else None
    distance_np = to_numpy(distance)
    arg_max_np = np.asarray(arg_max) if arg_max is not None else None
    p_z_np = p_z_val if p_z_val is not None else None
    
    print('Rec min max', np.min(rec_z_np), np.max(rec_z_np))
    return reconstructions_np, rec_z_np,z_prime_np,z_np,title_info,distance_np,arg_max_np,loss_val,log_pred_val,p_z_np,label_switch_step,pred_np


def create_counterfactual_data(plot_path,num_imgs,input_data,bb_box_coordinates,cnn_spn_model,
                               model_name,opt_weights,learning_rate = 0.01,num_steps = 150,replicates = 50):
    all_data = []

    for image_idx in range(num_imgs):
        print('image index', image_idx)
        x = np.asarray([input_data[0][image_idx]])
        y = np.asarray([input_data[1][image_idx][0]])
        add_info_data = np.asarray([input_data[1][image_idx][1:]])
        if len(bb_box_coordinates):
            coordinates = bb_box_coordinates[image_idx]
        else:
            coordinates=[]

        X = np.repeat(x, replicates, axis=0)
        additional_info = np.repeat(add_info_data, replicates, axis=0)
        # handle tensor or numpy
        if not isinstance(X, torch.Tensor):
            pred = cnn_spn_model.model_execution_X(torch.tensor(X, dtype=torch.float32), training = False)
        else:
            pred = cnn_spn_model.model_execution_X(X, additional_info, training=False)
        #        pred = cnn_spn_model.model_execution_X(torch.tensor(X, dtype=torch.float32)) if not isinstance(X, torch.Tensor) else cnn_spn_model.model_execution_X(X)
        # numpy for argmax
        if torch.is_tensor(pred):
            pred_np = pred.detach().cpu().numpy()
        else:
            pred_np = np.asarray(pred)
        mean_cls=np.mean(np.argmax(pred_np,axis=-1))
        opposite_class = int((round(mean_cls)+ 1) % 2)
        # return reconstructions_np, rec_z_np,z_prime_np,z_np,title_info,distance_np,arg_max_np,loss_val,log_pred_val,p_z_np,label_switch_step,pred_np
        reconstructions, rec_z, z_prime, z, title_info, distance, arg_max, loss, log_pred, p_z,label_switch_step, pred_out = get_counterfactual_infos(
            image_idx, cnn_spn_model, X, additional_info, opposite_class, y,model_name, opt_weights, learning_rate=learning_rate, num_steps=num_steps)

        all_data.append((reconstructions, rec_z, title_info, distance, arg_max, loss, log_pred, p_z, opposite_class,
                         coordinates, x,mean_cls,y))
    # save data in tmp folder
    pkl.dump(all_data, open(plot_path, 'wb'))
    return all_data


def process_image(image_idxs, input_data, bb_box_coordinates, model_name, opt_weights, learning_rate,
                  num_steps, replicates,add_info, num_classes, input_shape, data_path_fold, params,model_path):
    print('load model',image_idxs[-1],flush=True)
    if model_name == 'MLP':
        cnn_spn_model = load_VAE_model(add_info, num_classes, input_shape, data_path_fold, params, path=model_path)
    else:
        cnn_spn_model= load_model(add_info, num_classes, input_shape, data_path_fold, params, path=model_path)
    print('model loaded', image_idxs[-1],flush=True)

    result=[]
    for image_idx in range(image_idxs[0],image_idxs[1]) :
        #print('Processing image index:', image_idx)
        x = np.asarray([input_data[0][image_idx]])
        y = np.asarray([input_data[1][image_idx][0]])
        add_info_data = np.asarray([input_data[1][image_idx][1:]])
        coordinates = bb_box_coordinates[image_idx] if len(bb_box_coordinates) else []

        X = np.repeat(x, replicates, axis=0)
        additional_info = np.repeat(add_info_data, replicates, axis=0)

        pred = cnn_spn_model.model_execution_X(X, additional_info, training=False)
        mean_cls = np.mean(np.argmax(pred.numpy(), axis=-1))
        opposite_class = int((round(mean_cls) + 1) % 2)
        result.append(get_counterfactual_infos(image_idx,
            cnn_spn_model, X, additional_info, opposite_class, y, model_name, opt_weights,
            learning_rate=learning_rate, num_steps=num_steps
        ) + (opposite_class, coordinates, x, mean_cls, y))


    return result



#





'''
def create_counterfactual_data_parallel(plot_path, num_imgs, input_data, bb_box_coordinates, cnn_spn_model, model_name, opt_weights, learning_rate=0.01, num_steps=150, replicates=50):
    print('NUM CPUs',mp.cpu_count())
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(
            process_image,
            [(i, input_data, bb_box_coordinates, cnn_spn_model, model_name, opt_weights, learning_rate, num_steps, replicates) for i in range(num_imgs)]
        )

    # Save results
    pkl.dump(results, open(plot_path, 'wb'))
    return results
'''


def create_counterfactual_data_parallel(plot_path, num_imgs, input_data, bb_box_coordinates, cnn_spn_model, model_name, opt_weights, learning_rate=0.01, num_steps=150, replicates=50):
    results = []
    print('NUM CPUs',mp.cpu_count())
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(
            process_image, i, input_data, bb_box_coordinates, cnn_spn_model, model_name, opt_weights, learning_rate, num_steps, replicates
        ) for i in range(num_imgs)]

        for future in futures:
            results.append(future.result())

    pkl.dump(results, open(plot_path, 'wb'))
    return results

def create_counterfactual_data_parallel_new(plot_path, num_imgs, input_data, bb_box_coordinates,  model_name, opt_weights,
                                            add_info, num_classes, input_shape, data_path_fold, params,model_path,
                                            learning_rate=0.01, num_steps=150, replicates=50):

    print('NUM CPUs',mp.cpu_count())
    #set_start_method("spawn", force=True)  # Fix issues with TensorFlow & multiprocessing

    all_data = []
    num_max_worker=1#10#25
    num_workers = min(num_imgs, num_max_worker)  # Use available CPU cores
    batch_size=num_imgs//num_max_worker
    image_idxs=[]

    for w in range(num_max_worker):
        image_idxs.append([batch_size*w,batch_size*(w+1)])
    print(num_workers,num_imgs,batch_size,image_idxs)

    # Use multiprocessing to process images in parallel
    start_time=time.time()

    with Pool(num_workers) as pool:
        results = pool.starmap(process_image, [
            (image_idx, input_data, bb_box_coordinates, model_name, opt_weights, learning_rate, num_steps, replicates,add_info, num_classes, input_shape, data_path_fold, params,model_path)
            for image_idx in image_idxs
        ])

    # Collect results
    all_data.extend(results)

    # Save data
    with open(plot_path, 'wb') as f:
        pkl.dump(all_data, f)
    print('time for one param combi',(time.time()-start_time)/60)
    return all_data




def make_individual_plot(image_num, betas, gammas, plot_path,save_file_path_plot,model_name,summarizing_fkt ,model_n,subtitle,use_background=True):
    fig_size=4
    print(gammas,betas)
    if  len(gammas)==1:
        fig, axis = plt.subplots(len(gammas),len(betas), figsize=(fig_size * len(betas), fig_size * 1))
    else:
        fig, axis = plt.subplots( len(gammas),len(betas), figsize=(fig_size*len(betas), fig_size*len(gammas)))
    last_img=None
    all_axis=[]
    '''
    if use_background:
        all_plot_data=[]
        for b_i, beta in enumerate(betas):
            for g_i, gamma in enumerate(gammas[:-1]):
                save_file_path = plot_path + 'datab' + str(b_i) + 'g' + str(g_i) + model_name + '.pkl'
                all_data = pkl.load(open(save_file_path, 'rb'))
                data = all_data[image_num]
                reconstructions, rec_z, title_info, distance, arg_max, loss, log_pred, p_z, opposite_class, coordinates, x = data
                differences_mean = summarizing_fkt(rec_z, reconstructions)
                all_plot_data.append(differences_mean)

        all_plot_data=np.asarray(all_plot_data)
        vmin=np.min(all_plot_data)
        vmax = np.max(all_plot_data)
        print('vmin,vmax',vmin,vmax)
    '''
    for b_i, beta in enumerate(betas):
        for g_i, gamma in enumerate(gammas):
            #save_file_path = plot_path + 'datab' + str(b_i) + 'g' + str(g_i) + model_name + '_1000.pkl'
            save_file_path = plot_path + 'datab' + str(beta).replace('.', '-') + 'g' + str(gamma).replace('.',
                                                                                                          '-') + model_name + '_1000.pkl'
            all_data_list = pkl.load(open(save_file_path, 'rb'))
            all_data = []
            for entry in all_data_list:
                all_data.extend(entry)
            #data = all_data[image_num]

            #reconstructions, rec_z, title_info, distance, arg_max, loss, log_pred, p_z, opposite_class,coordinates, x, mean_cls, y=data
            #reconstructions, rec_z, title_info, distance, arg_max, loss, log_pred, p_z, opposite_class, coordinates, x = data
            #x_cf, x_rec, z_cf, z, title_info, distance, arg_max, loss, log_pred, p_z, label_switch_step, y_cf_goal,coordinates, x_org, mean_cls, y_org = data#zip(*all_data)
            [x_cf, x_rec, z_cf, z, title_info, distance, arg_max, loss, log_pred, p_z, label_switch_step, y_cf_goal, coordinates, x_org, mean_cls, y_org] = zip(*all_data)
            #print(np.min(x_cf), np.max(x_cf), np.min(x_org), np.max(x_org), np.min(x_rec), np.max(x_rec),x_cf.shape)
            x_cf = np.asarray(x_cf)
            x_rec = np.asarray(x_rec)
            x_org=np.asarray(x_org)
            coordinates=np.asarray(coordinates)[image_num]

            differences_mean = summarizing_fkt(x_cf[image_num], x_rec[image_num])
            '''

            arg_max_mean = np.mean(arg_max, axis=0)
            loss_mean = np.mean(loss, axis=0)
            distance_mean = np.mean(distance, axis=0)
            log_pred_mean = np.mean(arg_max, axis=0)
            p_z_mean = np.mean(p_z, axis=0)
            '''


            mini_title ='beta:' + str(beta) + ' gamma:' + str(gamma) #+ "\ny_np:{:.2f},y_cf{:.0f};L:{:.2f};MSE{:.2f};\nlog(p(y|z)):{:.2f},p(z):{:.2f}".format(
                                                                      ##                                 opposite_class,arg_max_mean,
                                                                       #                                loss_mean,
                                                                       #                                distance_mean,
                                                                       #                                log_pred_mean,
                                                                       #                                p_z_mean)
            if len(gammas)>1:
                ax = axis[g_i, b_i]
            else:
                ax = axis[ b_i]
            if use_background:
                ax.imshow(x_org[image_num][0, :, :, 0], cmap='gray')

                last_img=ax.imshow(differences_mean, cmap='jet', alpha=0.5)#,vmax=vmax,vmin=vmin)
                # Add a colorbar specific to this subplot
                fig.colorbar(last_img, ax=ax, orientation='vertical')
            else:
                last_img = ax.imshow(differences_mean, cmap='gray')
            ax.set_title(mini_title)
            if len(coordinates):
                x_co, y_co, w_co, h_co = coordinates
                factor = 128 / 1024
                rect = patches.Rectangle((x_co * factor, y_co * factor), w_co * factor, h_co * factor, linewidth=2,
                                         edgecolor='r',
                                         facecolor='none')
                ax.add_patch(rect)
            ax.axis('off')
            all_axis.append(ax)
        # Add a single shared colorbar
    '''
    if use_background:
        cbar = fig.colorbar(last_img, ax=all_axis, location='right', shrink=0.8)
        cbar.set_label('Intensity')
    '''
    fig.suptitle(subtitle + ' model:' + str(model_name))
    #fig.suptitle(model_n+' y'+str(y))
    plt.tight_layout()
    print('save',save_file_path_plot)
    plt.show()
    plt.savefig(save_file_path_plot)

def make_raw_plot(image_num, betas, gammas, plot_path,save_file_path_plot,model_name,summarizing_fkt ,model_n,subtitle,use_background=False):
    fig_size=4
    print(gammas,betas)
    if  len(gammas)==1:
        fig, axis = plt.subplots(len(gammas),len(betas), figsize=(fig_size * len(betas), fig_size * 1))
    else:
        fig, axis = plt.subplots( len(gammas),len(betas), figsize=(fig_size*len(betas), fig_size*len(gammas)))
    last_img=None
    all_axis=[]

    for b_i, beta in enumerate(betas):
        for g_i, gamma in enumerate(gammas):
            save_file_path = plot_path + 'datab' + str(beta).replace('.', '-') + 'g' + str(gamma).replace('.',
                                                                                                          '-') + model_name + '_1000.pkl'
            all_data_list = pkl.load(open(save_file_path, 'rb'))
            all_data = []
            for entry in all_data_list:
                all_data.extend(entry)
            #data = all_data[image_num]
            #x_cf, x_rec, z_cf, z, title_info, distance, arg_max, loss, log_pred, p_z, label_switch_step, y_cf_goal,coordinates, x_org, mean_cls, y_org = data#zip(*all_data)
            [x_cf, x_rec, z_cf, z, title_info, distance, arg_max, loss, log_pred, p_z, label_switch_step, y_cf_goal,
             coordinates, x_org, mean_cls, y_org] = zip(*all_data)
            x_cf = np.asarray(x_cf)
            x_rec = np.asarray(x_rec)
            x_org=np.asarray(x_org)
            coordinates = np.asarray(coordinates)[image_num]

            differences_mean = np.mean(x_cf[image_num],axis=0)
            '''

            arg_max_mean = np.mean(arg_max, axis=0)
            loss_mean = np.mean(loss, axis=0)
            distance_mean = np.mean(distance, axis=0)
            log_pred_mean = np.mean(arg_max, axis=0)
            p_z_mean = np.mean(p_z, axis=0)
            '''


            mini_title ='beta:' + str(beta) + ' gamma:' + str(gamma) #+ "\ny_np:{:.2f},y_cf{:.0f};L:{:.2f};MSE{:.2f};\nlog(p(y|z)):{:.2f},p(z):{:.2f}".format(
                                                                      ##                                 opposite_class,arg_max_mean,
                                                                       #                                loss_mean,
                                                                       #                                distance_mean,
                                                                       #                                log_pred_mean,
                                                                       #                                p_z_mean)
            if len(gammas)>1:
                ax = axis[g_i, b_i]
            else:
                ax = axis[ b_i]
            if use_background:
                ax.imshow(x_org[image_num][0, :, :, 0], cmap='gray')

                #last_img=ax.imshow(differences_mean, cmap='jet', alpha=0.5)#,vmax=vmax,vmin=vmin)
                # Add a colorbar specific to this subplot
                fig.colorbar(last_img, ax=ax, orientation='vertical')
            else:
                last_img = ax.imshow(differences_mean, cmap='gray')
            ax.set_title(mini_title)
            if len(coordinates):
                x_co, y_co, w_co, h_co = coordinates
                factor = 128 / 1024
                rect = patches.Rectangle((x_co * factor, y_co * factor), w_co * factor, h_co * factor, linewidth=2,
                                         edgecolor='r',
                                         facecolor='none')
                ax.add_patch(rect)
            ax.axis('off')
            all_axis.append(ax)
        # Add a single shared colorbar
    '''
    if use_background:
        cbar = fig.colorbar(last_img, ax=all_axis, location='right', shrink=0.8)
        cbar.set_label('Intensity')
    '''
    fig.suptitle(subtitle + ' model:' + str(model_name))
    #fig.suptitle(model_n+' y'+str(y))
    plt.tight_layout()
    print('save',save_file_path_plot)
    plt.show()
    plt.savefig(save_file_path_plot)


def individual_plot(num_plots, save_fig_name, all_data,beta,gamma,model_name,subtitle,summarizing_fkt):
    fig, axis = plt.subplots(num_plots, num_plots, figsize=(16, 16))

    for i in range(num_plots):
        for j in range(num_plots):
            data = all_data[i * num_plots + j]
            reconstructions, rec_z, title_info, distance, arg_max, loss, log_pred, p_z, opposite_class,coordinates, x, mean_cls, y=data
            #reconstructions, rec_z, title_info, distance, arg_max, loss, log_pred, p_z, opposite_class, coordinates, x = data



            differences_mean =summarizing_fkt(rec_z,reconstructions)

            arg_max_mean = np.mean(arg_max, axis=0)
            loss_mean = np.mean(loss, axis=0)
            distance_mean = np.mean(distance, axis=0)
            log_pred_mean = np.mean(arg_max, axis=0)
            p_z_mean = np.mean(p_z, axis=0)
            if len(coordinates):
                x_co, y_co, w_co, h_co = coordinates
                factor = 128 / 1024
                rect = patches.Rectangle((x_co * factor, y_co * factor), w_co * factor, h_co * factor, linewidth=2,
                                         edgecolor='r',
                                         facecolor='none')

            mini_title = "y_org:{:.0f},y_np{:.0f},y_cf:{:.2f};L:{:.2f}\nMSE{:.2f};log(p(y|z)):{:.2f},p(z):{:.2f}".format(y,
                                                                                                       opposite_class,arg_max_mean,
                                                                                                       loss_mean,
                                                                                                       distance_mean,
                                                                                                       log_pred_mean,
                                                                                                       p_z_mean)

            ax = axis[i, j]
            ax.imshow(x[0, :, :, 0], cmap='gray')
            ax.imshow(differences_mean, cmap='jet', alpha=0.5)
            ax.set_title(mini_title)
            ax.add_patch(rect)
            ax.axis('off')
    fig.suptitle(subtitle+':beta' + str(beta) + 'gamma' + str(gamma) + 'model' + str(model_name))
    plt.savefig(save_fig_name)
    #plt.show()

def diff_counterfactual_z(rec_z,reconstructions):
    # x_repeat = np.repeat(x[:, :, :, 0], reconstructions.shape[0], axis=0)
    reconstructions_int = np.asarray(reconstructions) #* 255
    org_rec_z = np.asarray(rec_z) #* 255
    differences = org_rec_z - reconstructions_int
    differences_mean = np.mean(differences, axis=0)
    return differences_mean

def mean_x_cf(rec_z,reconstructions):
    reconstructions_int = np.asarray(reconstructions) #* 255
    differences_mean = np.mean(reconstructions_int, axis=0)
    return differences_mean
def mean_x_org(rec_z,reconstructions):
    reconstructions_int = np.asarray(rec_z) #* 255
    differences_mean = np.mean(reconstructions_int, axis=0)
    return differences_mean

def one_x_cf(rec_z,reconstructions):
    reconstructions_int = np.asarray(reconstructions) #* 255
    return reconstructions_int[0]
def one_x_org(rec_z,reconstructions):
    reconstructions_int = np.asarray(rec_z)
    return reconstructions_int[0]
def diff_alternative_z(rec_z,reconstructions):
    # x_repeat = np.repeat(x[:, :, :, 0], reconstructions.shape[0], axis=0)
    reconstructions_int = np.asarray(reconstructions) #* 255
    org_rec_z = np.asarray(rec_z) #* 255

    std_c_x=np.std(reconstructions_int,axis=0)
    std_o_x = np.std(org_rec_z, axis=0)
    # normalize and reverse weights

    #w_std_c_x=1-(std_c_x/np.max(std_c_x,keepdims=True))
    w_std_o_x = 1 - (std_o_x/ np.max(std_o_x, keepdims=True))

    #w_std_c_x=1-(np.tanh(((std_c_x/np.max(std_c_x,keepdims=True))-0.5)*5)+0.5)
    #w_std_o_x = 1-(np.tanh(((std_o_x/ np.max(std_o_x, keepdims=True))-0.5)*5)+0.5)




    differences = org_rec_z - reconstructions_int
    differences_mean = np.mean(differences, axis=0)*w_std_o_x#*w_std_c_x
    return differences_mean
def std_z(rec_z,reconstructions):
    reconstructions_int = np.asarray(reconstructions) #* 255
    org_rec_z = np.asarray(rec_z) #* 255
    #differences = org_rec_z - reconstructions_int
    differences_mean = np.std(org_rec_z, axis=0)
    return differences_mean
def std_counterfactual(rec_z,reconstructions):
    reconstructions_int = np.asarray(reconstructions) #* 255
    org_rec_z = np.asarray(rec_z) #* 255
    #differences = org_rec_z - reconstructions_int
    differences_mean = np.std(reconstructions_int, axis=0)
    return differences_mean

def multiple_vanilla_counterfactuals(dataset_name, params, data_path_grid, add_info,model_n,
                                     num_imgs=5,path='../',model_path='../',learning_rate = 0.01,num_steps = 150,fold_idx=0,replicates =50):


    train, test, num_classes = load_dataset(dataset_name, binary=False,
                                            load_net=params.load_pretrain_model,
                                            machine=params.machine, path=path)
    input_data=test

    data_path_fold = data_path_grid + 'fold_' + str(fold_idx) + '/'
    bb_box_coordinates=[]
    input_shape = input_data[0].shape[1:]


    num_imgs = min(num_imgs,input_data[0].shape[0])
    print('num_imgs,input_data[0].shape[0]',num_imgs,input_data[0].shape[0])
    plot_path = data_path_fold + 'counterfactual_imgs/'
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)
    # 0. load model
    for model_name in ['SPN','MLP']:
        betas = [1,0]
        if model_name=='MLP':

            gammas=[0]

        else:

            gammas=[1,0]


        for b_i,beta in enumerate(betas):
            for g_i,gamma in enumerate(gammas):
                save_file_path=plot_path+'datab'+str(beta).replace('.','-')+'g'+str(gamma).replace('.','-')+model_name+'_1000_new.pkl'
                print('beta',beta,'gamma', gamma,'model',model_name,flush=True)
                if not os.path.isfile(save_file_path):
                    time_start=time.time()
                    all_data=create_counterfactual_data_parallel_new(save_file_path,num_imgs,input_data,
                                                        bb_box_coordinates,model_name,[beta,gamma],
                                                                     add_info, num_classes, input_shape, data_path_fold, params,model_path,
                                                                     learning_rate = learning_rate,num_steps = num_steps,replicates=replicates)
                    print('Time to evaluate',num_imgs,'images:',(time.time()-time_start)//60)








