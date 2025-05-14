import numpy as np


def rescale_to_01(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(arr)


def mean_cf(x_cf_,additional_data,argmax_arr_):
    x_cf=np.copy(x_cf_)
    argmax_arr=np.copy(argmax_arr_)
    x_cf_mean = np.mean(x_cf, axis=1)
    arg_max_mean = np.mean(argmax_arr, axis=1)
    return x_cf_mean,arg_max_mean

def max_cf(x_cf, additional_data,argmax_arr):
    #TODO this works only for y_cf_goal =1 - the general case has still to be implemented!

    arg_max = np.argmax(additional_data, axis=1)  # Get the max indices along axis 1
    x_cf_mean = x_cf[np.arange(x_cf.shape[0]), arg_max]  # Use advanced indexing
    arg_max_mean =  argmax_arr[np.arange(x_cf.shape[0]), arg_max]
    return x_cf_mean,arg_max_mean

def min_loss(x_cf, additional_data,argmax_arr):
    arg_max = np.argmin(additional_data, axis=1)  # Get the max indices along axis 1
    x_cf_mean = x_cf[np.arange(x_cf.shape[0]), arg_max]  # Use advanced indexing
    arg_max_mean =argmax_arr[np.arange(x_cf.shape[0]), arg_max]
    return x_cf_mean,arg_max_mean
def weighted(x_cf, additional_data,argmax_arr):
    # create weights in the area [0,1]
    rescale=rescale_to_01(additional_data)
    arg_max_mean = np.sum(argmax_arr*rescale, axis=1)
    upscale=np.expand_dims(np.expand_dims(np.expand_dims(rescale,axis=-1),axis=-1),axis=-1)
    x_cf_mean = np.sum(x_cf*upscale, axis=1)

    return x_cf_mean,arg_max_mean