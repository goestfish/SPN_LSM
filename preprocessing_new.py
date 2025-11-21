import os
import random
import pickle
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from PIL import Image


NIH_ROOT = "data" 
CSV_PATH = os.path.join(NIH_ROOT, "Data_Entry_2017.csv")
IMAGE_ROOT = NIH_ROOT

OUTPUT_PKL = "chexpert.pkl"
RANDOM_SEED = 42



def process_image(image_path, size=(128, 128)):
    """
    Load an image from disk, convert to grayscale,
    apply CLAHE, resize, and return as numpy array.
    """
    # Load with PIL
    pil_img = Image.open(image_path).convert("RGB")
    img = np.array(pil_img)

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)

    # Resize to target size
    resized_img = cv2.resize(clahe_img, size, interpolation=cv2.INTER_AREA)

    # Normalize to [0,1] as float32

    return resized_img  # shape: (H, W)


def load_and_filter_metadata():
    df = pd.read_csv(CSV_PATH)

    # Keep only Cardiomegaly or No Finding
    mask = (df["Finding Labels"] == "Cardiomegaly") | (df["Finding Labels"] == "No Finding")
    df = df[mask].copy()

    # Keep only frontal images (PA or AP)
    df = df[df["View Position"].isin(["PA", "AP"])].copy()

    # Map labels: 0 = No Finding, 1 = Cardiomegaly
    def label_fn(labels):
        if labels == "No Finding":
            return 0
        elif labels == "Cardiomegaly":
            return 1
        else:
            raise ValueError(f"Unexpected label: {labels}")

    df["label"] = df["Finding Labels"].apply(label_fn)

    # Basic cleaning / encoding
    # Sex: 'M' / 'F' -> 0 / 1 (or you can flip if you want)
    df["Sex_code"] = df["Patient Gender"].map({"M": 0, "F": 1})

    # View Position: PA=0, AP=1
    df["View_code"] = df["View Position"].map({"PA": 0, "AP": 1})

    # Age: just use as float
    df["Age"] = df["Patient Age"].astype(float)

    # Build full image path
    # Most Kaggle NIH dumps have flat images folder; if not, adjust here
    image_map = build_image_map(IMAGE_ROOT)

    df["image_path"] = df["Image Index"].map(image_map)

    # Drop any rows where we couldn't find the file (should be few / none)
    df = df.dropna(subset=["image_path"])

    return df


def build_image_map(root):
    """
    Walk through all subfolders under `root` and build
    a dict: { '00000001_000.png': '/full/path/to/images_001/00000001_000.png', ... }
    """
    image_map = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".png"):
                full_path = os.path.join(dirpath, fname)
                image_map[fname] = full_path
    return image_map


def patient_level_split(df, train_ratio=0.8, val_ratio=0.2, seed=RANDOM_SEED):
    """
    Split patients into train/val/test (patient-level).
    Train+Val = train_ratio, Test = 1 - train_ratio.
    Val is val_ratio * Train.
    """
    patient_ids = df["Patient ID"].unique().tolist()
    random.seed(seed)
    random.shuffle(patient_ids)

    num_patients = len(patient_ids)
    train_end = int(train_ratio * num_patients)

    train_patients_tmp = patient_ids[:train_end]
    test_patients = patient_ids[train_end:]

    # Split train into train/val
    random.seed(seed + 1)
    random.shuffle(train_patients_tmp)
    val_end = int(len(train_patients_tmp) * val_ratio)
    val_patients = train_patients_tmp[:val_end]
    train_patients = train_patients_tmp[val_end:]

    def assign_split(pid):
        if pid in train_patients:
            return "train"
        elif pid in val_patients:
            return "validation"
        else:
            return "test"

    df["split"] = df["Patient ID"].apply(assign_split)
    return df



def balance_per_split(df):
    """
    For each split, downsample majority class so we have ~50/50.
    """
    balanced_dfs = {}
    for split_name in ["train", "validation", "test"]:
        split_df = df[df["split"] == split_name].copy()

        cardio = split_df[split_df["label"] == 1]
        no_find = split_df[split_df["label"] == 0]

        n_cardio = len(cardio)
        n_no_find = len(no_find)
        print(f"{split_name}: {n_cardio} cardiomegaly, {n_no_find} no finding")

        if n_cardio == 0 or n_no_find == 0:
            print(f"⚠️ Warning: one class empty in {split_name}, skipping balancing")
            balanced_dfs[split_name] = split_df
            continue

        target_size = min(n_cardio, n_no_find)

        cardio_bal = cardio.sample(n=target_size, random_state=RANDOM_SEED)
        no_find_bal = no_find.sample(n=target_size, random_state=RANDOM_SEED)

        balanced = pd.concat([cardio_bal, no_find_bal]).sample(frac=1.0, random_state=RANDOM_SEED)
        balanced_dfs[split_name] = balanced

        print(f"{split_name} balanced: {len(balanced)} total ({target_size} + {target_size})")

    return balanced_dfs



def build_numpy_and_save(balanced_dfs, save_path):
    save_data = {}

    for split_name, split_df in balanced_dfs.items():
        X_list = []
        int_list = []
        float_list = []

        for _, row in split_df.iterrows():
            img_path = row["image_path"]
            img = process_image(img_path)  # (128,128) float32

            X_list.append(img)

            # y, sex_code, view_code
            y = int(row["label"])
            sex_code = int(row["Sex_code"])
            view_code = int(row["View_code"])

            int_list.append([y, sex_code, view_code])
            float_list.append([row["Age"]])

        X = np.stack(X_list)  # (N, 128, 128)
        int_data = np.stack(int_list)  # (N, 3)
        float_data = np.stack(float_list)  # (N, 1)

        print(f"{split_name}: X shape = {X.shape}, int_data shape = {int_data.shape}, float_data shape = {float_data.shape}")
        print(f"{split_name}: cardiomegaly percentage = {np.mean(int_data[:, 0])}")

        save_data[split_name] = [X, int_data, float_data]

    with open(save_path, "wb") as f:
        pickle.dump(save_data, f)

    print(f"Saved to {save_path}")



if __name__ == "__main__":
    # 1) Load & filter metadata
    df = load_and_filter_metadata()
    print("After filtering:")
    print(df["Finding Labels"].value_counts())

    # 2) Patient-level split
    df = patient_level_split(df)

    # 3) Balance 50/50 per split
    balanced = balance_per_split(df)

    # 4) Build numpy arrays & save as pickle
    build_numpy_and_save(balanced, OUTPUT_PKL)
