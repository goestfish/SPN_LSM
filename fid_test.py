import pickle
import numpy as np
from evaluation.metrics import fid

with open("chexpert.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data["train"][0][:64]
Y_train = data["train"][1][:64]
X_val = data["validation"][0][:64]
Y_val = data["validation"][1][:64]

fid_basic = fid.compute_fid(X_train, X_val)
print("Basic FID (train vs val):", float(fid_basic))

y_org = (Y_train[:, 0] > 0.5).astype(int)
y_cf_result = (Y_val[:, 0] > 0.5).astype(int)
y_cf_expected_goal = y_cf_result.copy()

rng = np.random.default_rng(0)
X_rec_like = X_train + rng.normal(0, 0.01, X_train.shape)
X_rec_like = np.clip(X_rec_like, 0, 255 if X_train.max() > 1 else 1)

rel_class_fid = fid.compute_class_fid(
    dataset1=X_train,
    dataset2=X_val,
    dataset3=X_rec_like,
    y_cf_expected_goal=y_cf_expected_goal,
    y_org=y_org,
    y_cf_result=y_cf_result
)
print("Relative class-FID (org vs rec vs cf):", float(rel_class_fid))