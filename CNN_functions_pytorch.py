import random
import psutil
import GPUtil
import sys
import time
import gc
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn import metrics as sk_metrics  # kept for compatibility, even if unused

from VAE import Resnet_VAE, make_pretrain_encoder  # assumed PyTorch implementation


# -------------------------------------------------------------------------
# Utility: memory usage (CPU + GPU)
# -------------------------------------------------------------------------

def get_memory_usage():
    """Returns the current CPU and GPU memory usage in GB."""
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / (1024 ** 3)

    gpus = GPUtil.getGPUs()
    gpu_memory = sum(gpu.memoryUsed for gpu in gpus) if gpus else 0

    return cpu_memory, gpu_memory


# -------------------------------------------------------------------------
# Torch Dataset for CNN / VAE
# -------------------------------------------------------------------------

class ChestDataset(Dataset):
    """
    Wraps numpy arrays into a PyTorch Dataset.

    X:   numpy array, shape (N, H, W, C) or (N, H, W)
    meta: numpy array, shape (N, >=1), where meta[:,0] is the label.
          Remaining columns are auxiliary info (sex, AP/PA, age, etc.).
    """

    def __init__(self, X, meta, add_info=True):
        if X.ndim == 3:  # (N, H, W)
            X = X[..., None]  # add channel dimension

        # (N, H, W, C) -> (N, C, H, W)
        self.images = torch.from_numpy(X).float().permute(0, 3, 1, 2)
        self.labels = torch.from_numpy(meta[:, 0]).long()

        self.add_info = add_info and meta.shape[1] > 1
        if self.add_info:
            self.other = torch.from_numpy(meta[:, 1:]).float()
        else:
            self.other = None

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]
        if self.add_info and self.other is not None:
            return x, y, self.other[idx]
        else:
            return x, y


# -------------------------------------------------------------------------
# Optional discretizer layer (not used by CNN but kept for compatibility)
# -------------------------------------------------------------------------

class Discretizer_layer(nn.Module):
    def __init__(self, initial_value=1.0, input_shape=16, dtype=torch.float32):
        super().__init__()
        init = torch.full((input_shape,), float(initial_value), dtype=dtype)
        self.bin = nn.Parameter(init)

    def forward(self, x):
        # x: (..., input_shape)
        # returns 1 where x > bin, else 0
        # broadcast self.bin to match x's shape on last dimension
        return (x > self.bin).float()


# -------------------------------------------------------------------------
# CNN definition (replacement for create_CNN)
# -------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self, grid_params, input_shape, num_classes, add_info):
        super().__init__()
        num_layer = grid_params.num_layer
        dropout = grid_params.dropout
        filter_size = grid_params.filter_size
        # Use latent_dim as last_num_filters to mirror TF code
        last_num_filters = grid_params.latent_dim if hasattr(grid_params, "latent_dim") else grid_params.end_dim_enc

        # input_shape from end_to_end_train is (H, W, C) or (C, H, W).
        # We'll assume CNN sees (C, H, W).
        if len(input_shape) == 3:
            if input_shape[0] in [1, 3]:
                in_channels, h, w = input_shape
            else:
                # assume H, W, C
                h, w, in_channels = input_shape
        else:
            raise ValueError(f"Unexpected input_shape: {input_shape}")

        layers = []
        in_ch = in_channels

        # Convolutional blocks (all but last)
        for lay_num in range(num_layer - 1):
            out_ch = 16 * (2 ** lay_num)
            layers.append(nn.Dropout(p=dropout / (2 ** lay_num)))
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=filter_size, padding=filter_size // 2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.BatchNorm2d(out_ch))
            in_ch = out_ch

        # Final conv block (no pooling yet)
        layers.append(nn.Dropout(p=dropout / (2 ** num_layer)))
        layers.append(nn.Conv2d(in_ch, last_num_filters, kernel_size=filter_size, padding=filter_size // 2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(last_num_filters))

        self.features = nn.Sequential(*layers)

        # determine spatial size after conv stack
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            feat = self.features(dummy)
            _, c, h_f, w_f = feat.shape

        # global pooling over remaining spatial extent
        self.pool = nn.MaxPool2d(kernel_size=(h_f, w_f))
        self.classifier = nn.Linear(last_num_filters, num_classes)

    def forward(self, x, return_embedding=False):
        feat = self.features(x)           # (N, C, H, W)
        pooled = self.pool(feat)          # (N, C, 1, 1)
        emb = pooled.view(pooled.size(0), -1)  # (N, C)
        logits = self.classifier(emb)     # (N, num_classes)
        if return_embedding:
            return emb, logits
        return logits


def create_CNN(grid_params, input_shape, num_classes, add_info):
    """
    PyTorch replacement for the original create_CNN.
    """
    model = SimpleCNN(grid_params, input_shape, num_classes, add_info)
    print('num trainable params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


# -------------------------------------------------------------------------
# CNN loading & evaluation (replacement for load_CNN)
# -------------------------------------------------------------------------

def _build_loaders_from_numpy(train_data, test_data, val_data, add_info, batch_size):
    X_train, meta_train = train_data
    X_test, meta_test = test_data
    X_val, meta_val = val_data

    train_ds = ChestDataset(X_train, meta_train, add_info=add_info)
    test_ds = ChestDataset(X_test, meta_test, add_info=add_info)
    val_ds = ChestDataset(X_val, meta_val, add_info=add_info)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader


def load_CNN(grid_params, input_shape, num_classes, debugging, ckpt, manager,
             checkpoint_path, train_data, test_data, val_data, add_info):
    """
    PyTorch version of load_CNN.
    ckpt and manager arguments are kept for API compatibility, but not used.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if grid_params.load_pretrain_model:
        model = load_pretrain_costume(grid_params, input_shape, num_classes)
    else:
        model = create_CNN(grid_params, input_shape, num_classes, add_info)

    model.to(device)

    learning_rate = grid_params.fine_tune_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # load checkpoint if exists
    os.makedirs(checkpoint_path, exist_ok=True)
    ckpt_file = os.path.join(checkpoint_path, 'cnn.pt')
    if os.path.exists(ckpt_file):
        state = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print(f"Restored CNN from {ckpt_file}")

    if debugging:
        first_param = next(model.parameters())
        print('DEBUGGING CNN weight sample', first_param.view(-1)[:5].detach().cpu().numpy())

    train_loader, test_loader, val_loader = _build_loaders_from_numpy(
        train_data, test_data, val_data, add_info, grid_params.batch_size
    )

    def eval_loader(loader):
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                if add_info and len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)
                total_loss += loss.item() * images.size(0)

                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        total_loss /= len(loader.dataset)
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        acc = accuracy_score(all_labels, all_preds)
        return total_loss, acc

    train_loss, train_acc = eval_loader(train_loader)
    test_loss, test_acc = eval_loader(test_loader)
    val_loss, val_acc = eval_loader(val_loader)

    print('CNN train loss', train_loss, 'train acc', train_acc,
          'test loss', test_loss, 'test acc', test_acc, flush=True)

    # keep return signature: model, ckpt, manager, metrics
    return model, None, None, [[train_loss, train_acc],
                               [val_loss, val_acc],
                               [test_loss, test_acc]]


# -------------------------------------------------------------------------
# VAE helpers (assume Resnet_VAE is a PyTorch module with similar API)
# -------------------------------------------------------------------------

def load_VAE(params, add_info, num_classes, input_shape, checkpoint_path, path=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Resnet_VAE(
        filter_size=params.filter_size,
        num_layer=params.num_layer,
        input_shape=input_shape,
        batchnorm_integration=params.batchnorm_integration,
        shortcut=params.shortcut,
        activation=params.activation,
        num_filter_encoder=params.num_filter_encoder,
        strides_encoder=params.strides_encoder,
        num_filter_decoder=params.num_filter_decoder,
        strides_decoder=params.strides_decoder,
        latent_dim=params.latent_dim,
        end_dim_enc=params.end_dim_enc,
        learning_rate=params.learning_rate,
        semi_supervised=params.semi_supervised,
        num_classes=num_classes,
        dropout=params.dropout,
        load_pretrain_model=params.load_pretrain_model,
        add_info=add_info,
        loss_weights=params.loss_weights,
        VAE_fine_tune=params.VAE_fine_tune,
        path=path,
        use_KLD_anneal=params.use_KLD_anneal
    )

    model.to(device)

    os.makedirs(checkpoint_path, exist_ok=True)
    ckpt_file = os.path.join(checkpoint_path, 'vae.pt')
    if os.path.exists(ckpt_file):
        state = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(state['model'])
        print(f"Restored VAE from {ckpt_file}")

    return model


def load_VAE_and_eval(params, input_shape, num_classes, debugging, ckpt, manager,
                      checkpoint_path, train_dataset, test_dataset, val_dataset, add_info):
    """
    PyTorch version of load_VAE_and_eval.
    Assumes train_dataset, test_dataset, val_dataset are DataLoaders.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_VAE(params, add_info, num_classes, input_shape, checkpoint_path)

    if debugging:
        first_param = next(model.parameters())
        print('DEBUGGING show example VAE weight sample',
              first_param.view(-1)[:5].detach().cpu().numpy())

    def eval_loader(loader):
        model.eval()
        total_losses = None
        num_rounds = 0
        with torch.no_grad():
            for batch in loader:
                if add_info and len(batch) == 3:
                    x, y, other = batch
                else:
                    x, y = batch
                    other = None
                x = x.to(device)
                y = y.to(device)
                if other is not None:
                    other = other.to(device)

                # expected API from Resnet_VAE: model.evaluate_batch returns
                # (loss, rec_loss, kld, clf_loss, mae, mse) or similar
                losses = model.evaluate_batch(x, y, other)
                losses_np = [float(l) for l in losses]
                if total_losses is None:
                    total_losses = np.zeros(len(losses_np), dtype=np.float64)
                total_losses += np.array(losses_np)
                num_rounds += 1

        if total_losses is None:
            return [0.0]
        total_losses /= max(num_rounds, 1)
        return total_losses.tolist()

    train_losses = eval_loader(train_dataset)
    test_losses = eval_loader(test_dataset)
    val_losses = eval_loader(val_dataset)

    return model, None, None, [[train_losses], [val_losses], [test_losses]]


# -------------------------------------------------------------------------
# Pretrained backbone integration (replacement for load_pretrain_costume)
# -------------------------------------------------------------------------

def load_pretrain_costume(grid_params, input_shape, num_classes):
    """
    Placeholder for integrating a pretrained backbone in PyTorch.
    In the TF version this used EfficientNetB7; here you can load any
    torchvision model (e.g. resnet18) and attach a small head.
    For now, we just use SimpleCNN as a stand-in.
    """
    print("WARNING: load_pretrain_costume currently uses SimpleCNN as a placeholder.")
    model = SimpleCNN(grid_params, input_shape, num_classes, add_info=False)
    return model


# -------------------------------------------------------------------------
# CNN training (replacement for train_embedding_cnn)
# -------------------------------------------------------------------------

def train_embedding_cnn(grid_params, val_data, input_shape,
                        num_classes, checkpoint_path, load_data, split,
                        train_data, add_info, acc_stop=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, meta_train = train_data
    X_val, meta_val = val_data

    train_ds = ChestDataset(X_train, meta_train, add_info=add_info)
    val_ds = ChestDataset(X_val, meta_val, add_info=add_info)

    train_loader = DataLoader(train_ds, batch_size=grid_params.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=grid_params.batch_size, shuffle=False)

    if grid_params.load_pretrain_model:
        model = load_pretrain_costume(grid_params, input_shape, num_classes)
    else:
        model = create_CNN(grid_params, input_shape, num_classes, add_info)

    model.to(device)

    learning_rate = grid_params.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if num_classes == 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    os.makedirs(checkpoint_path, exist_ok=True)
    ckpt_file = os.path.join(checkpoint_path, f'cnn_split{split}.pt')

    if acc_stop:
        best_val_metric = 0.0
    else:
        best_val_metric = float('inf')

    if load_data and os.path.exists(ckpt_file):
        state = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print(f"Restored CNN from {ckpt_file}")

    for epoch in range(grid_params.epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            if add_info and len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            if num_classes == 1:
                labels_f = labels.float().view(-1, 1)
                loss = criterion(logits, labels_f)
            else:
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                if add_info and len(batch) == 3:
                    inputs, labels, _ = batch
                else:
                    inputs, labels = batch

                inputs = inputs.to(device)
                labels = labels.to(device)

                logits = model(inputs)
                if num_classes == 1:
                    labels_f = labels.float().view(-1, 1)
                    loss = criterion(logits, labels_f)
                else:
                    loss = criterion(logits, labels)
                val_loss += loss.item() * inputs.size(0)

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        val_loss /= len(val_loader.dataset)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        if num_classes == 1:
            preds = all_logits.view(-1).detach().numpy()
            labels_np = all_labels.detach().numpy()
            val_acc = -sk_metrics.mean_squared_error(labels_np, preds)
        else:
            preds = all_logits.argmax(dim=1).numpy()
            labels_np = all_labels.numpy()
            val_acc = accuracy_score(labels_np, preds)

        print(f"Epoch {epoch+1}/{grid_params.epochs} "
              f"- train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}",
              flush=True)

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
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       ckpt_file)
            first_param = next(model.parameters())
            print('Saved CNN weights sample', first_param.view(-1)[:5].detach().cpu().numpy(), flush=True)

    # keep original return signature (no ckpt/manager objects in PyTorch version)
    return model, None, None


# -------------------------------------------------------------------------
# VAE training helpers (validate_data, beta_scheduler, load_model, train_embedding_VAE)
# -------------------------------------------------------------------------

def validate_data(val_dataset, add_info, model, beta=0):
    """
    Evaluate VAE on a validation DataLoader.
    Expects model.execute_net_xy(x, y, other, training=False, beta=beta)
    to return: (loss, rec_loss, kld, clf, other_losses, pred, z, extra)
    """
    device = next(model.parameters()).device
    losses = np.zeros((4 + 2), dtype=np.float64)
    num_rounds = 0
    predictions = []
    all_z = []
    gt = []

    model.eval()
    with torch.no_grad():
        for batch in val_dataset:
            if add_info and len(batch) == 3:
                test_x, test_y, test_other = batch
            else:
                test_x, test_y = batch
                test_other = None

            test_x = test_x.to(device)
            test_y = test_y.to(device)
            if test_other is not None:
                test_other = test_other.to(device)

            loss, rec_loss, kld, clf, other_losses, pred, z, _ = model.execute_net_xy(
                test_x, test_y, test_other, training=False, beta=beta
            )

            losses[0] += float(loss)
            losses[1] += float(rec_loss)
            losses[2] += float(kld)
            losses[3] += float(clf)
            predictions.extend(pred.cpu().numpy().tolist())
            gt.extend(test_y.cpu().numpy().tolist())

            for i, tmp_loss in enumerate(other_losses):
                losses[4 + i] += float(tmp_loss.mean())

            all_z.append([z.cpu().numpy(), test_y.cpu().numpy().tolist()])
            num_rounds += 1

    return all_z, predictions, gt, losses, num_rounds


def beta_scheduler(epoch, total_epochs, epoch_steps=100, max_beta=1.0):
    # Linear annealing, like original
    return min(max_beta, (epoch % epoch_steps) / epoch_steps)


def load_model(params, num_classes, add_info, checkpoint_path, checkpoint_path_tmp,
               le_warmup, input_shape, encoder, init=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Resnet_VAE(
        filter_size=params.filter_size,
        num_layer=params.num_layer,
        input_shape=input_shape,
        batchnorm_integration=params.batchnorm_integration,
        shortcut=params.shortcut,
        activation=params.activation,
        num_filter_encoder=params.num_filter_encoder,
        strides_encoder=params.strides_encoder,
        num_filter_decoder=params.num_filter_decoder,
        strides_decoder=params.strides_decoder,
        latent_dim=params.latent_dim,
        end_dim_enc=params.end_dim_enc,
        learning_rate=params.learning_rate,
        semi_supervised=params.semi_supervised,
        num_classes=num_classes,
        dropout=params.dropout,
        load_pretrain_model=params.load_pretrain_model,
        add_info=add_info,
        loss_weights=params.loss_weights,
        VAE_fine_tune=params.VAE_fine_tune,
        use_GAN=params.GAN,
        use_KLD_anneal=params.use_KLD_anneal,
        le_warmup=le_warmup,
        gauss_std=params.gauss_std,
        encoder=encoder
    )

    model.to(device)

    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(checkpoint_path_tmp, exist_ok=True)
    ckpt_file = os.path.join(checkpoint_path, 'vae.pt')
    ckpt_tmp_file = os.path.join(checkpoint_path_tmp, 'vae_tmp.pt')

    if init and os.path.exists(ckpt_file):
        state = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(state['model'])
        print(f"Restored VAE from {ckpt_file}")
        torch.save({'model': model.state_dict()}, ckpt_tmp_file)
    elif not init and os.path.exists(ckpt_tmp_file):
        state = torch.load(ckpt_tmp_file, map_location=device)
        model.load_state_dict(state['model'])
        print(f"Restored temp VAE from {ckpt_tmp_file}")
    else:
        # fresh model, save initial temp
        torch.save({'model': model.state_dict()}, ckpt_tmp_file)

    # mimic original return: model, manager, ckpt, manager_tmp, ckpt_tmp
    return model, None, None, None, None


def train_embedding_VAE(params, train_data, test_data, val_data, input_shape,
                        num_classes, checkpoint_path, load_data, split,
                        add_info, checkpoint_path_tmp, acc_stop=True):
    """
    PyTorch version of train_embedding_VAE.

    Expects Resnet_VAE with:
      - attributes: semi_supervised
      - methods:
          * train_semi(train_loader, beta)
          * train(train_loader)
          * evaluate_(loader, verbose)
          * execute_net_xy(...)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = None
    if params.load_pretrain_model:
        encoder = make_pretrain_encoder(
            params.filter_size,
            num_layer=0,
            input_shape=input_shape,
            batchnorm_integration=params.batchnorm_integration,
            num_filter=0,
            shortcut=0,
            strides=0,
            activation=0,
            encoder_name=0,
            dtype=0,
            dilations=0,
            dropout=params.dropout,
            end_dim=params.end_dim_enc,
            path=''
        )

    model, manager, ckpt, manager_tmp, ckpt_tmp = load_model(
        params, num_classes, add_info, checkpoint_path, checkpoint_path_tmp,
        params.le_warmup, input_shape, encoder[0] if encoder is not None else None, init=True
    )

    def make_loaders(randomize=False):
        X_train, meta_train = train_data
        X_test, meta_test = test_data
        X_val, meta_val = val_data

        train_ds = ChestDataset(X_train, meta_train, add_info=add_info)
        test_ds = ChestDataset(X_test, meta_test, add_info=add_info)
        val_ds = ChestDataset(X_val, meta_val, add_info=add_info)

        train_loader = DataLoader(train_ds, batch_size=params.batch_size,
                                  shuffle=randomize or True)
        test_loader = DataLoader(test_ds, batch_size=params.batch_size,
                                 shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=params.batch_size,
                                shuffle=False)
        return train_loader, test_loader, val_loader

    train_loader, test_loader, val_loader = make_loaders(randomize=False)

    all_val_losses = []
    all_debugging_stuff = []
    if acc_stop:
        best_val_metric = 0.0
    else:
        print('stop for best loss on L')
        best_val_metric = float('inf')

    epochs = params.epochs

    # initial val loss (use model.evaluate_ if implemented)
    if hasattr(model, "evaluate_"):
        init_val_losses = model.evaluate_(val_loader, verbose=0)
        val_loss = init_val_losses[0] if isinstance(init_val_losses, (list, tuple, np.ndarray)) else init_val_losses
    else:
        val_loss = 0.0
    print('VAE initial val loss:', val_loss)

    for epoch in range(epochs):
        # re-create loaders with randomization to mimic TF behavior
        del train_loader, val_loader
        gc.collect()
        train_loader, _, val_loader = make_loaders(randomize=True)

        cpu_mem_before, gpu_mem_before = get_memory_usage()
        print(f"Epoch {epoch + 1}/{epochs} "
              f"- CPU: {cpu_mem_before:.2f} GB | GPU: {gpu_mem_before:.2f} GB",
              flush=True)

        beta = beta_scheduler(epoch, epochs)
        start_time_train = time.time()

        if getattr(model, "semi_supervised", False):
            # expected: model.train_semi(train_loader, beta) -> loss, rec_loss, kld, clf_loss, mae, mse
            loss, rec_loss, kld, clf_loss, mae, mse = model.train_semi(train_loader, beta=beta)
            train_loss = [loss, rec_loss, kld, clf_loss, mae, mse]
            print("Train Semi | Epoch: {:03d} | Loss: {:.4f} | Rec Loss: {:.4f} | "
                  "KLD: {:.4f} | Clf Loss: {:.4f} | MAE: {:.4f} | MSE: {:.4f}".format(
                      epoch, loss, rec_loss, kld, clf_loss, mae, mse),
                  flush=True)

            if np.isnan(loss):
                print('Loss is NaN, restoring previous model snapshot (tmp).')
                tmp_file = os.path.join(checkpoint_path_tmp, 'vae_tmp.pt')
                if os.path.exists(tmp_file):
                    state = torch.load(tmp_file, map_location=device)
                    model.load_state_dict(state['model'])
                continue

            if epoch % 10 == 0 and not np.isnan(loss):
                print('Start VAE validation', flush=True)

                all_z, predictions, gt, losses_arr, num_rounds = validate_data(val_loader, add_info, model, beta=beta)
                all_z_train, predictions_train, gt_train, losses_train, num_rounds_train = validate_data(
                    train_loader, add_info, model, beta=beta
                )

                if num_rounds > 0:
                    losses_arr = losses_arr / num_rounds
                losses_arr = np.round(losses_arr, 3)
                val_loss = losses_arr[0]
                mae_val = losses_arr[-2]

                # classification accuracy
                if len(predictions) > 0:
                    preds_np = np.array(predictions)
                    softmax_pred = torch.softmax(torch.from_numpy(preds_np), dim=-1).numpy()
                    arg_max = np.argmax(softmax_pred, axis=-1)
                    acc_loss = accuracy_score(gt, arg_max, normalize=True)
                    balanced_acc = balanced_accuracy_score(gt, arg_max)
                else:
                    acc_loss = 0.0
                    balanced_acc = 0.0

                rec = 0
                if params.VAE_debug:
                    # use small batch for reconstruction debug
                    test_x = test_data[0][:16]
                    train_x = train_data[0][:16]
                    if add_info:
                        test_y = test_data[1][:16, 0]
                        train_y = train_data[1][:16, 0]
                        test_add_info = test_data[1][:16, 1:]
                        train_add_info = train_data[1][:16, 1:]
                    else:
                        test_y = test_data[1][:16]
                        train_y = train_data[1][:16]
                        test_add_info = None
                        train_add_info = None

                    test_x_t = torch.from_numpy(test_x).float().permute(0, 3, 1, 2).to(device)
                    train_x_t = torch.from_numpy(train_x).float().permute(0, 3, 1, 2).to(device)
                    test_y_t = torch.from_numpy(test_y).long().to(device)
                    train_y_t = torch.from_numpy(train_y).long().to(device)
                    if test_add_info is not None:
                        test_add_t = torch.from_numpy(test_add_info).float().to(device)
                        train_add_t = torch.from_numpy(train_add_info).float().to(device)
                    else:
                        test_add_t = None
                        train_add_t = None

                    out_rec = model.execute_net_xy(test_x_t, test_y_t, test_add_t, training=False, beta=beta)
                    out_rec_train = model.execute_net_xy(train_x_t, train_y_t, train_add_t, training=False, beta=beta)

                    z, gt_vec = zip(*all_z)
                    z = np.concatenate(z, axis=0)
                    gt_vec = np.concatenate(gt_vec, axis=0)

                    z_train, gt_train_vec = zip(*all_z_train)
                    z_train = np.concatenate(z_train, axis=0)
                    gt_train_vec = np.concatenate(gt_train_vec, axis=0)

                    rec = out_rec[-1].cpu().numpy()
                    all_debugging_stuff.append([
                        [losses_arr, z, gt_vec, rec, predictions],
                        [train_loss, z_train, gt_train_vec, out_rec_train[-1].cpu().numpy(), predictions_train]
                    ])
                    size_in_bytes = sys.getsizeof(all_debugging_stuff)
                    size_in_mb = size_in_bytes / (1024 ** 2)
                    print(f"Memory used by all_debugging_stuff: {size_in_mb:.2f} MB")

                    del test_x_t, train_x_t, test_y_t, train_y_t

                print(
                    "Val Results Semi | Epoch: {:03d} | Acc Loss: {:.4f} | Balanced Acc: {:.4f} | "
                    "MAE: {:.4f} | min_rec: {:.4f} | max_rec: {:.4f}".format(
                        epoch, acc_loss, balanced_acc, mae_val,
                        float(np.min(rec)) if np.size(rec) else 0.0,
                        float(np.max(rec)) if np.size(rec) else 0.0
                    ),
                    flush=True,
                )

                improve = False
                if acc_stop:
                    if balanced_acc > best_val_metric:
                        best_val_metric = balanced_acc
                        improve = True
                else:
                    if val_loss < best_val_metric:
                        best_val_metric = val_loss
                        improve = True

                if improve:
                    ckpt_file = os.path.join(checkpoint_path, 'vae.pt')
                    torch.save({'model': model.state_dict()}, ckpt_file)
                    print('Saved VAE weights', flush=True)

                print(
                    "VAE Val Loss: {:.4f} | Acc Loss: {:.4f} | Improve: {} | Time: {:.2f} min | Beta: {:.4f}".format(
                        val_loss, acc_loss, improve, (time.time() - start_time_train) / 60, beta
                    ),
                    flush=True,
                )
                all_val_losses.append(losses_arr)
        else:
            # non-semi-supervised mode (not commonly used in paper)
            loss, rec_loss, kld, mae, mse = model.train(train_loader)
            print(
                "Train | Epoch: {:03d} | Loss: {:.4f} | Rec Loss: {:.4f} | "
                "KLD: {:.4f} | MAE: {:.4f} | MSE: {:.4f}".format(
                    epoch, loss, rec_loss, kld, mae, mse
                ),
                flush=True,
            )

            # validation loop for non-semi-supervised mode
            model.eval()
            losses_arr = np.zeros((3 + 2), dtype=np.float64)
            num_rounds = 0
            with torch.no_grad():
                for batch in val_loader:
                    if add_info and len(batch) == 3:
                        test_x, y, test_other = batch
                    else:
                        test_x, y = batch
                        test_other = None

                    test_x = test_x.to(device)
                    if test_other is not None:
                        test_other = test_other.to(device)

                    loss_v, rec_loss_v, kld_v, other_losses = model.execute_net(
                        test_x, training=False
                    )
                    losses_arr[0] += float(loss_v)
                    losses_arr[1] += float(rec_loss_v)
                    losses_arr[2] += float(kld_v)
                    for i, tmp_loss in enumerate(other_losses):
                        losses_arr[3 + i] += float(tmp_loss.mean())

                    num_rounds += 1

            if num_rounds > 0:
                losses_arr = losses_arr / num_rounds
            val_loss = losses_arr[0]
            losses_arr = np.round(losses_arr, 3)

            print('VAE val results', epoch, losses_arr, flush=True)
            improve = False
            if val_loss < best_val_metric:
                best_val_metric = val_loss
                improve = True
                ckpt_file = os.path.join(checkpoint_path, 'vae.pt')
                torch.save({'model': model.state_dict()}, ckpt_file)
                print('Saved VAE weights', flush=True)

            print('VAE val loss:', val_loss, 'improve', improve,
                  'time:', (time.time() - start_time_train) / 60, flush=True)
            all_val_losses.append(losses_arr)

        # save tmp snapshot each epoch
        ckpt_tmp_file = os.path.join(checkpoint_path_tmp, 'vae_tmp.pt')
        torch.save({'model': model.state_dict()}, ckpt_tmp_file)
        print('Epoch time', (time.time() - start_time_train) / 60, flush=True)

    return model, None, None, all_val_losses, all_debugging_stuff


# -------------------------------------------------------------------------
# Embedding extraction (replacement for get_layer_embeddings)
# -------------------------------------------------------------------------

def get_layer_embeddings(grid_params, model, layer_names, all_datasets, get_max=True, add_info=False):
    """
    PyTorch version of get_layer_embeddings.

    In the original code:
      - if grid_params.use_VAE: intermediate_layer_model = model.clf_model()
      - else: intermediate_layer_model = keras.Model(inputs=model.input, outputs=layer_output)

    Here:
      - if use_VAE: assume model.clf_model() returns a module whose forward gives embedding maps
      - else: assume CNN model has forward(..., return_embedding=True) that returns (embedding, logits)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('START EMBEDDING', layer_names)

    if getattr(grid_params, "use_VAE", 0):
        embedding_model = model.clf_model().to(device)
    else:
        embedding_model = model.to(device)

    output_embeds = []
    all_labels = [[] for _ in range(len(all_datasets))]
    all_other_info = [[] for _ in range(len(all_datasets))]
    num_embedding_layers = 1  # we only support one embedding for now

    for ds_idx, input_dataset in enumerate(all_datasets):
        all_batch_embeddings = [[] for _ in range(num_embedding_layers)]
        for ds_record in input_dataset:
            if add_info and len(ds_record) == 3:
                images, labels, other = ds_record
                all_other_info[ds_idx].append(other.numpy())
            else:
                images, labels = ds_record
                other = None

            images = images.to(device)

            with torch.no_grad():
                if getattr(grid_params, "use_VAE", 0):
                    # assume clf_model() forward returns a list/tuple with embeddings as first element
                    prediction_0 = embedding_model(images)[0]
                else:
                    emb, _ = embedding_model(images, return_embedding=True)
                    prediction_0 = emb  # (N, D)

            all_labels[ds_idx].append(labels.numpy())

            if len(layer_names) > 1:
                for lay in range(num_embedding_layers):
                    if get_max and prediction_0.dim() == 4:
                        diff = prediction_0[:, lay].amax(dim=(1, 2)).cpu().numpy()
                    else:
                        diff = prediction_0.cpu().numpy()
                    all_batch_embeddings[lay].extend(diff)
            else:
                if get_max and prediction_0.dim() == 4:
                    diff = prediction_0.amax(dim=(2, 3)).cpu().numpy()
                else:
                    diff = prediction_0.cpu().numpy()
                all_batch_embeddings[0].extend(diff)

        all_labels[ds_idx] = np.concatenate(all_labels[ds_idx], axis=0)
        if add_info and all_other_info[ds_idx]:
            all_other_info[ds_idx] = np.concatenate(all_other_info[ds_idx], axis=0)

        curr_embeds = [np.stack(entry, axis=0) for entry in all_batch_embeddings]
        if len(curr_embeds) > 1:
            concat_embeds = np.concatenate(curr_embeds, axis=-1)
            output_embeds.append(concat_embeds)
        else:
            output_embeds.append(curr_embeds[0])
            print(curr_embeds[0].shape)

    return output_embeds, all_labels, all_other_info
