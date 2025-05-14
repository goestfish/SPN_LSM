import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
from skimage.measure import label, regionprops
from scipy.spatial.distance import pdist
#import tensorflow_gan.python.eval.inception_metrics as inception
#import tensorflow_hub as hub

# Load InceptionV3 model from TF-Hub
#FID_INCEPTION = 'https://tfhub.dev/google/tfgan/eval/inception/1'
##inception_model = hub.load(FID_INCEPTION)

'''
def preprocess_images(images, image_size=299):
    """Preprocess images to match InceptionV3 requirements."""
    images = tf.image.resize(images, (image_size, image_size))  # Resize
    images = (images - 127.5) / 127.5  # Normalize to [-1, 1]
    return images

def get_inception_activations(images):
    """Extract activations from the InceptionV3 model."""
    images = preprocess_images(images)
    return inception_model(images)

'''


# Function to preprocess images for Inception
def preprocess_images(images, image_size=299):
    """Resizes and normalizes images for InceptionV3."""
    images = tf.image.resize(images, (image_size, image_size))  # Resize
    images = (images*2)-1#(images - 127.5) / 127.5  # Normalize to [-1, 1]
    return images

# Function to compute FID
def compute_fid(dataset1, dataset2):
    """Computes FID score between two datasets using TF-GAN."""
    #print(np.min(dataset1),np.max(dataset1))
    #print(np.min(dataset2),np.max(dataset2))
    dataset1 = preprocess_images(dataset1)
    dataset2 = preprocess_images(dataset2)
    #print(np.min(dataset1),np.max(dataset1))
    #print(np.min(dataset2),np.max(dataset2))

    dataset1=np.repeat(dataset1,repeats=3,axis=-1)
    dataset2 = np.repeat(dataset2, repeats=3, axis=-1)

    fid_score = tfgan.eval.frechet_inception_distance(dataset1, dataset2)
    return fid_score.numpy()  # Convert tensor to NumPy


def split_by_class(X, y,unique_classes):
    #tmp=np.where(y == 0)
    split_data = {cls: X[np.where(y == cls)[0]] for cls in unique_classes}
    return split_data

def compute_class_fid(dataset1, dataset2,dataset3,y_cf_expected_goal, y_org,y_cf_result):
    y_old=np.mod(y_cf_expected_goal+1,2)
    class_labels=np.unique(y_org)
    ds_org=split_by_class(dataset1,y_org,class_labels)
    ds_cf=split_by_class(dataset2,y_cf_result,class_labels)
    ds_rec=split_by_class(dataset3,y_old,class_labels)

    rel_FID=0

    for class_lab in class_labels:
        partial_org=ds_org[class_lab]
        patrial_cf=ds_cf[class_lab]
        patrial_rec=ds_rec[class_lab]

        fid_rec=compute_fid(partial_org,patrial_rec)
        fid_cf = compute_fid(partial_org, patrial_cf)
        rel_FID+=(fid_rec-fid_cf)


    rel_FID=rel_FID/len(class_labels)


    return rel_FID


def compute_rel_L2(dataset1, dataset2,dataset3):
    L2_rec = compute_MSE(dataset1, dataset3)
    L2_cf = compute_MSE(dataset1, dataset2)
    rel_L2 = L2_rec[1] - L2_cf[1]
    return rel_L2

def get_importance_region_information(x_org,x_cf_strat,threshold=0.5):
    diff=x_org-x_cf_strat

    # Calculate thresholds
    threshold_pos = 0.5 * np.max(diff)
    threshold_neg = 0.5 * np.min(diff)

    # Generate masks for positive and negative changes
    mask_pos = diff > threshold_pos
    mask_neg = diff < threshold_neg

    # Label connected components in each mask
    #labeled_pos = label(mask_pos)
    #labeled_neg = label(mask_neg)

    ######################
    # Connected Component Analysis
    result_arr=[[],[]]
    for bin_idx,(binary_mask, title) in enumerate(zip([mask_pos,mask_neg],['label pos','label neg'])):
        for idx in range(x_org.shape[0]):
            labeled_mask = label(binary_mask[idx,:,:,0])
            regions = regionprops(labeled_mask)

            # Analyze regions
            blob_sizes = [region.area for region in regions]
            blob_count = len(blob_sizes)
            if len(blob_sizes):
                result_arr[bin_idx].append([sum(blob_sizes)/len(blob_sizes),blob_count])

    #######################

    #result_arr=np.asarray(result_arr)
    result=np.zeros((2,2))
    for e_idx,entry in enumerate(result_arr):
        entry = np.asarray(entry)
        entry=np.mean(entry,axis=0)
        result[e_idx]=entry
    return result


def compute_MSE(dataset1, dataset2):
    dataset1=np.squeeze(dataset1)
    dataset2 = np.squeeze(dataset2)
    sqr=np.square(dataset1 - dataset2)
    mae=np.mean(sqr, axis=(1,2))
    L2=np.sqrt(np.sum(sqr,axis=(1,2)))
    L2_mean=np.mean(L2)
    mae_mean=np.mean(mae)
    return mae_mean,L2_mean


def compute_MSE_mean(dataset1, dataset2):
    dataset1=np.squeeze(dataset1)
    dataset2 = np.squeeze(dataset2)
    dataset1=np.expand_dims(dataset1,1)
    sqr=np.square(dataset1 - dataset2)
    mae=np.mean(sqr, axis=(2,3))
    L2=np.sqrt(np.sum(sqr,axis=(2,3)))
    L2_mean=np.mean(np.mean(L2,axis=1))
    mae_mean=np.mean(mae)
    return mae_mean,L2_mean


def compute_MAE(dataset1, dataset2):
    dataset1=np.squeeze(dataset1)
    dataset2 = np.squeeze(dataset2)
    abs=np.abs(dataset1 - dataset2)
    mae=np.mean(abs, axis=(1,2))
    L1=np.sum(abs,axis=(1,2))
    L1_mean=np.mean(L1)
    mae_mean=np.mean(mae)
    return mae_mean,L1_mean



def compute_MAE_mean(dataset1, dataset2):
    dataset1=np.squeeze(dataset1)
    dataset2 = np.squeeze(dataset2)
    dataset1=np.expand_dims(dataset1,1)
    abs=np.abs(dataset1 - dataset2)
    mae=np.mean(abs, axis=(2,3))
    L1=np.sum(abs,axis=(2,3))
    L1_mean=np.mean(np.mean(L1,axis=1))
    mae_mean=np.mean(mae)
    return mae_mean,L1_mean


def compute_validity(expected_y_cf,predicted_y_cf):
    #expected_y_cf=np.argmax(expected_y_cf)
    #predicted_y_cf=np.argmax(predicted_y_cf)
    same=(expected_y_cf==predicted_y_cf)
    validity=np.mean(same)
    return validity



def mean_lsp_calculation(latent_representations,metric='euclidean'):

    N, k, d = latent_representations.shape
    diversity_scores = []

    for i in range(N):
        reps = latent_representations[i]  # shape (k, d)
        if k < 2:
            diversity_scores.append(0.0)
        else:
            pairwise_dists = pdist(reps, metric=metric)
            avg_dist = np.mean(pairwise_dists)
            diversity_scores.append(avg_dist)

    return np.mean(diversity_scores)



'''

if __name__ == '__main__':
    # test:
    # Example: Two random datasets (Replace with real images)
    dataset1 = tf.random.uniform([100, 64, 64, 3], 0, 255, dtype=tf.float32)  # Fake dataset 1
    dataset2 = tf.random.uniform([100, 64, 64, 3], 0, 255, dtype=tf.float32)  # Fake dataset 2

    # Compute FID
    fid_value = compute_fid(dataset1, dataset2)
    print(f'FID Score: {fid_value}')
'''
