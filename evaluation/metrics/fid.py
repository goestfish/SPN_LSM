import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg
from skimage.measure import label, regionprops
from scipy.spatial.distance import pdist

def _to_3ch_299(x: np.ndarray) -> torch.Tensor:
    if x.ndim == 3:
        x = x[..., None]
    x = x.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    t = torch.from_numpy(x).permute(0, 3, 1, 2)
    t = t.repeat(1, 3, 1, 1)
    t = F.interpolate(t, size=(299, 299), mode="bilinear", align_corners=False)
    t = t * 2.0 - 1.0
    return t

@torch.no_grad()
def _inception_activations(x: np.ndarray, batch_size: int = 64, device: str = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True).to(device)
    model.eval()
    feats = []
    def _hook(_, __, output):
        feats.append(output.detach())
    handle = model.avgpool.register_forward_hook(_hook)
    x = _to_3ch_299(x).to(device)
    N = x.size(0)
    for i in range(0, N, batch_size):
        _ = model(x[i:i + batch_size])
    handle.remove()
    f = torch.cat(feats, dim=0)
    f = torch.flatten(f, 1)
    return f.cpu().numpy()

def _stats(feats: np.ndarray):
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma

def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # Ensure vector / matrix shapes
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    # If one covariance is scalar-like (1x1) and the other is DxD,
    # promote the scalar variance to an isotropic DxD covariance
    if sigma1.shape != sigma2.shape:
        if sigma1.shape == (1, 1) and sigma2.shape[0] == sigma2.shape[1]:
            var = float(sigma1[0, 0])
            sigma1 = np.eye(sigma2.shape[0]) * var
        elif sigma2.shape == (1, 1) and sigma1.shape[0] == sigma1.shape[1]:
            var = float(sigma2[0, 0])
            sigma2 = np.eye(sigma1.shape[0]) * var
        else:
            # Fallback: align to the smaller dimensionality (very defensive)
            d = min(sigma1.shape[0], sigma2.shape[0])
            sigma1 = sigma1[:d, :d]
            sigma2 = sigma2[:d, :d]

    diff = mu1 - mu2

    # Product of covariance matrices
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # If imaginary components appear due to numerical issues, drop them
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Add small epsilon on the diagonal if needed for numerical stability
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return float(
        diff.dot(diff)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2.0 * tr_covmean
    )

def compute_fid(dataset1: np.ndarray, dataset2: np.ndarray) -> float:
    f1 = _inception_activations(dataset1)
    f2 = _inception_activations(dataset2)
    mu1, sigma1 = _stats(f1)
    mu2, sigma2 = _stats(f2)
    return float(_frechet_distance(mu1, sigma1, mu2, sigma2))

def split_by_class(X: np.ndarray, y: np.ndarray, unique_classes):
    return {cls: X[np.where(y == cls)[0]] for cls in unique_classes}

def compute_class_fid(dataset1, dataset2, dataset3, y_cf_expected_goal, y_org, y_cf_result):
    """
    Compute class-wise relative FID (rec vs cf), averaged only over classes
    that have enough samples in *all* three datasets.
    """
    y_old = np.mod(y_cf_expected_goal + 1, 2)
    class_labels = np.unique(y_org)

    ds_org = split_by_class(dataset1, y_org, class_labels)
    ds_cf  = split_by_class(dataset2, y_cf_result, class_labels)
    ds_rec = split_by_class(dataset3, y_old, class_labels)

    rel_FID = 0.0
    valid_classes = 0

    for cls in class_labels:
        X_org = ds_org.get(cls, np.empty((0,)))
        X_cf  = ds_cf.get(cls,  np.empty((0,)))
        X_rec = ds_rec.get(cls, np.empty((0,)))

        # Skip classes where any dataset is empty or has too few samples
        if (
            X_org.size == 0 or X_cf.size == 0 or X_rec.size == 0 or
            X_org.shape[0] < 2 or X_cf.shape[0] < 2 or X_rec.shape[0] < 2
        ):
            continue

        fid_rec = compute_fid(X_org, X_rec)
        fid_cf  = compute_fid(X_org, X_cf)

        rel_FID += (fid_rec - fid_cf)
        valid_classes += 1

    if valid_classes == 0:
        # Nothing to average over: define 0.0 as "no signal"
        return 0.0

    rel_FID /= valid_classes
    return float(rel_FID)


def compute_MSE(dataset1, dataset2):
    dataset1 = np.squeeze(dataset1)
    dataset2 = np.squeeze(dataset2)
    sqr = np.square(dataset1 - dataset2)
    mae = np.mean(sqr, axis=(1, 2))
    L2 = np.sqrt(np.sum(sqr, axis=(1, 2)))
    return float(np.mean(mae)), float(np.mean(L2))

def compute_MSE_mean(dataset1, dataset2):
    dataset1 = np.squeeze(dataset1)
    dataset2 = np.squeeze(dataset2)
    dataset1 = np.expand_dims(dataset1, 1)
    sqr = np.square(dataset1 - dataset2)
    mae = np.mean(sqr, axis=(2, 3))
    L2 = np.sqrt(np.sum(sqr, axis=(2, 3)))
    return float(np.mean(mae)), float(np.mean(np.mean(L2, axis=1)))

def compute_MAE(dataset1, dataset2):
    dataset1 = np.squeeze(dataset1)
    dataset2 = np.squeeze(dataset2)
    absd = np.abs(dataset1 - dataset2)
    mae = np.mean(absd, axis=(1, 2))
    L1 = np.sum(absd, axis=(1, 2))
    return float(np.mean(mae)), float(np.mean(L1))

def compute_MAE_mean(dataset1, dataset2):
    dataset1 = np.squeeze(dataset1)
    dataset2 = np.squeeze(dataset2)
    dataset1 = np.expand_dims(dataset1, 1)
    absd = np.abs(dataset1 - dataset2)
    mae = np.mean(absd, axis=(2, 3))
    L1 = np.sum(absd, axis=(2, 3))
    return float(np.mean(mae)), float(np.mean(np.mean(L1, axis=1)))

def compute_validity(expected_y_cf, predicted_y_cf):
    same = (expected_y_cf == predicted_y_cf)
    return float(np.mean(same))

def mean_lsp_calculation(latent_representations, metric='euclidean'):
    N, k, d = latent_representations.shape
    diversity_scores = []
    for i in range(N):
        reps = latent_representations[i]
        if k < 2:
            diversity_scores.append(0.0)
        else:
            pairwise_dists = pdist(reps, metric=metric)
            diversity_scores.append(np.mean(pairwise_dists))
    return float(np.mean(diversity_scores))

def get_importance_region_information(x_org, x_cf_strat, threshold=0.5):
    diff = x_org - x_cf_strat
    thr_pos = 0.5 * np.max(diff)
    thr_neg = 0.5 * np.min(diff)
    mask_pos = diff > thr_pos
    mask_neg = diff < thr_neg
    res = [[], []]
    for bin_idx, binary_mask in enumerate([mask_pos, mask_neg]):
        for idx in range(x_org.shape[0]):
            labeled_mask = label(binary_mask[idx, :, :, 0])
            regions = regionprops(labeled_mask)
            blob_sizes = [r.area for r in regions]
            if blob_sizes:
                res[bin_idx].append([sum(blob_sizes) / len(blob_sizes), len(blob_sizes)])
    out = np.zeros((2, 2))
    for e_idx, entry in enumerate(res):
        if entry:
            entry = np.asarray(entry)
            out[e_idx] = np.mean(entry, axis=0)
    return out
