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

def min_loss(x_cf, additional_data, argmax_arr):
    """
    Pick, for each image, the counterfactual with minimum 'loss' in additional_data.

    x_cf:         (N, K, H, W, C) or (N, K, ...)
    additional_data: ideally (N, K), but may come as 1D
    argmax_arr:   (N, K) class indices per replicate
    """
    x_cf = np.asarray(x_cf)
    additional_data = np.asarray(additional_data)
    argmax_arr = np.asarray(argmax_arr)

    N = x_cf.shape[0]
    K = x_cf.shape[1] if x_cf.ndim >= 2 else 1

    # Ensure additional_data is 2D (N, K)
    if additional_data.ndim == 1:
        if additional_data.shape[0] == K:
            # loss per replicate (same for all images)
            additional_data = np.tile(additional_data[None, :], (N, 1))  # (N, K)
        elif additional_data.shape[0] == N:
            # loss per image, no per-replicate variation
            additional_data = additional_data[:, None]  # (N, 1)
        else:
            # fallback: uniform "loss" across replicates
            additional_data = np.zeros((N, K))
    elif additional_data.ndim == 2:
        # ok, but if second dim doesn't match K, try to fix
        if additional_data.shape[1] != K:
            # simplest fallback: repeat / crop to K
            if additional_data.shape[1] == 1:
                additional_data = np.tile(additional_data, (1, K))
            else:
                additional_data = additional_data[:, :K]
    else:
        # higher dims: flatten everything but batch and treat last dim as K
        additional_data = additional_data.reshape(N, -1)
        if additional_data.shape[1] != K:
            additional_data = additional_data[:, :K]

    # now (N, K)
    arg_max = np.argmin(additional_data, axis=1)  # (N,)

    # pick the chosen replicate for each image
    x_cf_mean = x_cf[np.arange(N), arg_max]
    arg_max_mean = argmax_arr[np.arange(N), arg_max]

    return x_cf_mean, arg_max_mean
def weighted(x_cf, additional_data, argmax_arr):
    """
    x_cf:        (N, K, ..., ..., ...)  e.g. (num_images, num_reps, 1, H, W)
    additional_data: used to build weights, can be (N, K), (N,), or (K,)
    argmax_arr:  (N, K)  argmax per image/replicate
    """
    x_cf = np.asarray(x_cf)
    argmax_arr = np.asarray(argmax_arr)

    # Create weights in [0, 1] from additional_data
    rescale = rescale_to_01(np.asarray(additional_data))

    N, K = argmax_arr.shape  # num images, num replicates

    # --- Force rescale to have shape (N, K) ---
    if rescale.shape == (N, K):
        pass  # already correct
    elif rescale.shape == (N,):
        # same weight for all K reps of each image
        rescale = np.repeat(rescale[:, None], K, axis=1)
    elif rescale.shape == (K,):
        # same weights across N images
        rescale = np.repeat(rescale[None, :], N, axis=0)
    else:
        # fallback: uniform weights over replicates
        rescale = np.ones((N, K), dtype=float) / float(K)

    # --- Aggregate over replicates for argmax ---
    arg_max_mean = np.sum(argmax_arr * rescale, axis=1)

    # --- Use the same weights on x_cf ---
    # rescale: (N, K) -> (N, K, 1, 1, 1) to match x_cf's trailing dims
    upscale = rescale[:, :, None, None, None]
    x_cf_mean = np.sum(x_cf * upscale, axis=1)

    return x_cf_mean, arg_max_mean
