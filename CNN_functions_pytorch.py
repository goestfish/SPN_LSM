# cnn_functions_torch.py  (or overwrite CNN_functions.py with this)

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from VAE import VAE, train_vae, test_vae
from utils import data_to_batch  # assumes you've converted this to return PyTorch DataLoaders


# ----------------------------
# Simple CNN backbone (optional)
# ----------------------------

class SimpleCNN(nn.Module):
    """
    Rough PyTorch analogue of your old Keras CNN:
    conv + maxpool stacks -> flatten -> linear classifier.
    """
    def __init__(self, grid_params, input_shape, num_classes):
        super().__init__()
        c_in = input_shape[-1]          # channels last in numpy
        h = input_shape[0]
        w = input_shape[1]

        layers = []
        num_layer = grid_params.num_layer
        dropout = grid_params.dropout
        filter_size = grid_params.filter_size
        last_num_filters = grid_params.last_num_filters

        in_ch = c_in
        for lay_num in range(num_layer - 1):
            num_filters = 16 * (2 ** lay_num)
            layers.append(nn.Dropout(dropout / (2 ** lay_num)))
            layers.append(nn.Conv2d(in_ch, num_filters,
                                    kernel_size=filter_size,
                                    padding="same"))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            layers.append(nn.BatchNorm2d(num_filters))
            in_ch = num_filters
            h //= 2
            w //= 2

        layers.append(nn.Dropout(dropout / (2 ** num_layer)))
        layers.append(nn.Conv2d(in_ch, last_num_filters,
                                kernel_size=filter_size,
                                padding="same"))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(last_num_filters))

        # global pool then dense
        layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        self.conv = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(last_num_filters, num_classes)

    def forward(self, x):
        # x: [B, H, W, C] -> [B, C, H, W]
        if x.dim() == 4 and x.shape[1] not in (1, 3):
            x = x.permute(0, 3, 1, 2).contiguous()
        z = self.conv(x)
        z = self.flatten(z)
        logits = self.classifier(z)
        return logits

    def get_feature_extractor(self):
        """
        For SPN structure learning we only need embeddings, not final logits.
        """
        class FeatureExtractor(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.conv = parent.conv

            def forward(self, x):
                if x.dim() == 4 and x.shape[1] not in (1, 3):
                    x = x.permute(0, 3, 1, 2).contiguous()
                z = self.conv(x)  # [B, C, 1, 1]
                return z
        return FeatureExtractor(self)


# ----------------------------
# CNN helpers (PyTorch)
# ----------------------------

def train_embedding_cnn(grid_params, val_data, input_shape,
                        num_classes, checkpoint_path, load_data,
                        split, train_data, add_info, acc_stop=True):
    """
    Simpler PyTorch version of your old train_embedding_cnn.
    It trains SimpleCNN and returns (model, manager_stub, ckpt_stub)
    so train_cnn_spn can keep the same call pattern.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(grid_params, input_shape, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=grid_params.learning_rate)

    if num_classes == 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # build loaders
    if add_info:
        X_val = val_data[0]
        y_val = val_data[1][:, 0].astype(np.int64)
    else:
        X_val = val_data[0]
        y_val = val_data[1].astype(np.int64)

    X_train = train_data[0]
    y_train = train_data[1][:, 0].astype(np.int64) if add_info else train_data[1].astype(np.int64)

    def make_loader(X, y):
        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).long()
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=grid_params.batch_size, shuffle=True)

    train_loader = make_loader(X_train, y_train)
    val_loader = make_loader(X_val, y_val)

    best_val_metric = -1e9 if acc_stop else 1e9

    for epoch in range(grid_params.epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        running_loss /= len(train_loader.dataset)

        # val
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / max(total, 1)
        print(f"[CNN] Epoch {epoch} train_loss={running_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        improved = False
        if acc_stop:
            if val_acc > best_val_metric:
                best_val_metric = val_acc
                improved = True
        else:
            if val_loss < best_val_metric:
                best_val_metric = val_loss
                improved = True
        if improved:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path + ".pt")

    # simple stubs for ckpt / manager to keep signature
    ckpt = None
    manager = None
    return model, manager, ckpt


def load_CNN(grid_params, input_shape, num_classes,
             debugging, ckpt, manager, checkpoint_path,
             train_data, test_data, val_data, add_info):
    """
    PyTorch version of load_CNN: build the same SimpleCNN, load state_dict
    if available, evaluate on train/val/test and return metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(grid_params, input_shape, num_classes).to(device)

    ckpt_path = checkpoint_path + ".pt"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[CNN] Restored from {ckpt_path}")

    if num_classes == 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    def eval_split(data):
        if add_info:
            X = data[0]
            y = data[1][:, 0].astype(np.int64)
        else:
            X = data[0]
            y = data[1].astype(np.int64)
        X_t = torch.from_numpy(X).float().to(device)
        y_t = torch.from_numpy(y).long().to(device)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=grid_params.batch_size, shuffle=False)

        model.eval()
        loss_total = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss_total += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        loss_total /= len(loader.dataset)
        acc = correct / max(total, 1)
        return loss_total, acc

    train_loss, train_acc = eval_split(train_data)
    val_loss, val_acc = eval_split(val_data)
    test_loss, test_acc = eval_split(test_data)

    print("[CNN] train_loss", train_loss, "train_acc", train_acc)
    print("[CNN] val_loss", val_loss, "val_acc", val_acc)
    print("[CNN] test_loss", test_loss, "test_acc", test_acc)

    MLP_eval = [[train_loss, train_acc], [val_loss, val_acc], [test_loss, test_acc]]
    return model, ckpt, manager, MLP_eval


# ----------------------------
# VAE helpers (PyTorch, using your new VAE)
# ----------------------------

def load_VAE(params, add_info, num_classes, input_shape, checkpoint_path):
    """
    Simple loader for your new PyTorch VAE.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = input_shape[-1]
    img_size = input_shape[0]

    model = VAE(in_channels=in_channels,
                latent_dim=params.latent_dim,
                img_size=img_size,
                num_classes=num_classes).to(device)

    ckpt_path = checkpoint_path + ".pt"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[VAE] Restored from {ckpt_path}")

    return model


def load_VAE_and_eval(params, input_shape, num_classes,
                      debugging, ckpt, manager, checkpoint_path,
                      train_dataset, test_dataset, val_dataset, add_info):
    """
    Mimics the old signature but in PyTorch:
    - loads VAE
    - evaluates train/val/test with test_vae()
    - returns MLP_eval in same nested [[...],[...],[...]] shape.
    train_dataset/test_dataset/val_dataset are assumed to be PyTorch DataLoaders
    created by utils.data_to_batch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_VAE(params, add_info, num_classes, input_shape, checkpoint_path).to(device)

    train_loss = test_vae(model, train_dataset, device)
    val_loss = test_vae(model, val_dataset, device)
    test_loss = test_vae(model, test_dataset, device)

    print(f"[VAE] train_loss={train_loss:.4f} val_loss={val_loss:.4f} test_loss={test_loss:.4f}")

    # keep shape [[train],[val],[test]]
    MLP_eval = [[train_loss], [val_loss], [test_loss]]

    # ckpt/manager: unused in torch path but keep for compatibility
    return model, ckpt, manager, MLP_eval


def train_embedding_VAE(params, train_data, test_data, val_data, input_shape,
                        num_classes, checkpoint_path, load_data, split,
                        add_info, checkpoint_path_tmp, acc_stop=True):
    """
    PyTorch VAE training wrapper that roughly replaces the old train_embedding_VAE.
    Uses your VAE.train_vae / test_vae, returns same tuple shape:
        model, manager_stub, ckpt_stub, all_val_losses, all_debugging_stuff
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = input_shape[-1]
    img_size = input_shape[0]

    model = VAE(in_channels=in_channels,
                latent_dim=params.latent_dim,
                img_size=img_size,
                num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # Turn raw numpy arrays into DataLoaders using your existing helper:
    train_ds, test_ds, val_ds = data_to_batch(
        train_data, test_data, val_data, add_info, params.batch_size
    )

    all_val_losses = []
    all_debugging_stuff = []

    best_val = 1e9
    for epoch in range(params.epochs):
        train_loss = train_vae(model, train_ds, optimizer, device)
        val_loss = test_vae(model, val_ds, device)
        print(f"[VAE] Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        all_val_losses.append(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path + ".pt")

    # no real "manager"/"ckpt" needed in torch version
    manager = None
    ckpt = None

    return model, manager, ckpt, all_val_losses, all_debugging_stuff


# ----------------------------
# Embedding extraction for SPN structure learning
# ----------------------------

def get_layer_embeddings(grid_params, model, layer_names, all_datasets,
                         get_max=True, add_info=False):
    """
    PyTorch version of get_layer_embeddings.

    For VAE:
        - we use model.clf_model() to get a latent extractor that returns mu.
    For CNN:
        - we use model.get_feature_extractor() to get conv feature maps.
    all_datasets is a list of PyTorch DataLoaders (train_dataset, test_dataset).
    Returns:
        output_embeds: [train_embed, test_embed] as numpy arrays
        all_labels:   [train_labels, test_labels]
        all_other_info: [train_other, test_other] (possibly empty arrays)
    """
    print("START EMBEDDING", layer_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if getattr(grid_params, "use_VAE", False):
        feature_model = model.clf_model().to(device)
    else:
        # assume our SimpleCNN or similar has get_feature_extractor
        feature_model = model.get_feature_extractor().to(device)

    feature_model.eval()

    output_embeds = []
    all_labels = [[] for _ in range(len(all_datasets))]
    all_other_info = [[] for _ in range(len(all_datasets))]

    num_embedding_layers = 1  # we only output one embedding per sample

    for ds_idx, loader in enumerate(all_datasets):
        all_batch_embeddings = [[] for _ in range(num_embedding_layers)]

        for batch in loader:
            if add_info:
                images, labels, other = batch
                all_other_info[ds_idx].append(other.cpu().numpy())
            else:
                images, labels = batch
            images = images.to(device)

            with torch.no_grad():
                feats = feature_model(images)  # [B, latent_dim] or [B, C, H, W]

            if feats.dim() == 4 and get_max:
                feats_np = feats.amax(dim=(2, 3)).cpu().numpy()
            else:
                feats_np = feats.cpu().numpy()

            all_batch_embeddings[0].extend(feats_np)
            all_labels[ds_idx].append(labels.cpu().numpy())

        all_labels[ds_idx] = np.concatenate(all_labels[ds_idx], axis=0)
        if add_info and len(all_other_info[ds_idx]) > 0:
            all_other_info[ds_idx] = np.concatenate(all_other_info[ds_idx], axis=0)
        else:
            all_other_info[ds_idx] = np.array([])

        curr_embeds = [np.stack(entry, axis=0) for entry in all_batch_embeddings]
        output_embeds.append(curr_embeds[0])
        print("Embedding shape dataset", ds_idx, curr_embeds[0].shape)

    return output_embeds, all_labels, all_other_info
