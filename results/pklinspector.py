import pandas as pd
import pickle as pkl
import numpy as np

# Load
data = pkl.load(open("full_run_100e_ft50.pkl", "rb"))
metrics_list = data[0]

# Convert to array
metrics_arr = np.array(metrics_list)  # shape: (folds, 2 phases, 3 splits, 7 metrics)

# We only care about test split (index 2)
# Metric index 0 = accuracy
rows = []
for fold_idx in range(metrics_arr.shape[0]):
    cnn_acc = metrics_arr[fold_idx, 0, 2, 0]      # before_cnnspn, test, acc
    spn_acc = metrics_arr[fold_idx, 1, 2, 0]      # after_cnnspn, test, acc
    rows.append([fold_idx, cnn_acc, spn_acc])

df = pd.DataFrame(rows, columns=["Fold", "CNN", "CNN + SPN"])
df = df.round(4)
df["Δ"] = (df["CNN + SPN"] - df["CNN"]).round(4)
print(df)
