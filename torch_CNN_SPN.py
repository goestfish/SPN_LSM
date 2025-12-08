import time
import random

import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from torch.utils.data import TensorDataset, DataLoader


def _to_tensor(x, dtype=torch.float32, device=None):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype, device=device)
    if torch.is_tensor(x):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


def create_embedding_model(input_dimensions):

    in_dim = int(np.prod(input_dimensions))

    class EmbeddingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, 10),
                nn.ReLU(),
                nn.Linear(10, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x)

    return EmbeddingModel()


class CNN_SPN(nn.Module):

    def __init__(self, num_classes, input_dimensions, learning_rate, spn=None, cnn=None, get_max=True):
        super(CNN_SPN, self).__init__()
        self.num_classes = num_classes
        self.get_max = get_max
        self.use_add_info = False

        if cnn is not None:
            self.embedding = cnn
        else:
            self.embedding = create_embedding_model(input_dimensions)

        if spn is not None:
            self.spn_training = True
            self.clf = spn
            self.clf_loss = None
        else:
            self.spn_training = False
            self.clf = nn.Sequential(
                nn.Linear(2, 1),
                nn.Sigmoid()
            )
            self.clf_loss = nn.BCELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def model_execution_X_y(self, X, y, other):
        device = next(self.parameters()).device
        X = _to_tensor(X, dtype=torch.float32, device=device)
        y = _to_tensor(y, dtype=torch.long, device=device)
        if other is not None and not isinstance(other, int):
            other = _to_tensor(other, dtype=torch.float32, device=device)
        else:
            other = None

        out = self.embedding(X)
        if isinstance(out, tuple):
            embedding = out[0]
        else:
            embedding = out

        if embedding.dim() > 2:
            if self.get_max:
                embedding = torch.amax(embedding, dim=tuple(range(1, embedding.dim())))
            else:
                embedding = embedding.view(embedding.size(0), -1)

        if self.spn_training:
            if self.use_add_info and other is not None:
                if other.dim() == 1:
                    other = other.unsqueeze(-1)
                embedding = torch.cat([embedding, other], dim=-1)

            y_float = y.to(torch.float32).unsqueeze(-1)
            spn_input = torch.cat([y_float, embedding], dim=-1)
            spn_output = self.clf(spn_input)
            loss = -torch.sum(spn_output)
        else:
            logits = self.clf(embedding).squeeze(-1)
            y_float = y.to(torch.float32)
            loss = self.clf_loss(logits, y_float)

        return loss

    def train_step(self, x, y, other):
        self.train()
        self.optimizer.zero_grad()
        loss = self.model_execution_X_y(x, y, other)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def train_model(self, train_ds, first_loss):
        all_losses = 0.0
        counter = 0.0
        for it_c, train_rec in enumerate(train_ds):
            other = 0
            if self.use_add_info:
                (X, y, other) = train_rec
            else:
                (X, y) = train_rec

            loss = self.train_step(X, y, other)
            all_losses += loss.item()
            counter += 1
            if not it_c and first_loss:
                print(it_c, 'loss', all_losses / counter)
        return all_losses / counter

    def eval_cnn(self, test_X):
        self.eval()
        with torch.no_grad():
            x = _to_tensor(test_X, dtype=torch.float32, device=next(self.parameters()).device)
            return self.embedding(x)

    def get_spn_variables(self):
        return list(self.clf.parameters()) if hasattr(self, "clf") else []


def enc_layer(net, stride, filter_num, batchnorm_integration, shortcut,
              dilations, filter_size, trainable, dtype, old_h, old_num_filter, dropout, activation,
              name):
    conv = nn.Conv2d(old_num_filter, filter_num, kernel_size=filter_size,
                     stride=stride, padding=filter_size // 2,
                     dilation=dilations, bias=False)
    layers = [conv]

    if batchnorm_integration:
        layers.append(nn.BatchNorm2d(filter_num))

    if dropout:
        layers.append(nn.Dropout2d(p=dropout))

    if activation is not None:
        layers.append(nn.ReLU() if activation is F.relu else nn.Tanh())

    block = nn.Sequential(*layers)
    net.append(block)
    return net, filter_num


def make_costume_encoder(filter_size, num_layer, input_shape, batchnorm_integration, num_filter, shortcut, strides,
                         activation, encoder_name, dtype, dilations, dropout, end_dim):
    class CostumeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            c_in = input_shape[2]
            img_size = input_shape[0]
            all_filters = [c_in] + num_filter
            blocks = []
            in_ch = c_in
            for layer_num, stride in enumerate(strides):
                out_ch = all_filters[layer_num + 1]
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=filter_size,
                                 stride=stride, padding=filter_size // 2,
                                 dilation=dilations, bias=False)
                sub_layers = [conv]
                if batchnorm_integration:
                    sub_layers.append(nn.BatchNorm2d(out_ch))
                if dropout:
                    sub_layers.append(nn.Dropout2d(p=dropout))
                if activation is not None:
                    if activation is F.relu:
                        sub_layers.append(nn.ReLU())
                    else:
                        sub_layers.append(nn.Tanh())
                blocks.append(nn.Sequential(*sub_layers))
                in_ch = out_ch
                img_size = img_size // stride

            self.blocks = nn.ModuleList(blocks)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(in_ch * img_size * img_size, end_dim)
            self.fc2 = nn.Linear(end_dim, 1)

        def forward(self, x):
            if x.dim() == 4 and x.shape[1] != 1 and x.shape[1] != 3:
                x = x.permute(0, 3, 1, 2).contiguous()
            for blk in self.blocks:
                x = blk(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

    return CostumeEncoder()


def gradient_penalty(critic, real_images, fake_images):
    device = real_images.device
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated.requires_grad_(True)

    critic_interpolates = critic(interpolated)
    if critic_interpolates.dim() > 1:
        critic_interpolates = critic_interpolates.view(-1)

    grads = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grads = grads.view(batch_size, -1)
    grad_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-8)
    penalty = torch.mean((grad_norm - 1.0) ** 2)
    return penalty


class CNN_SPN_Parts(nn.Module):
    def __init__(self,
                 num_classes,
                 learning_rate,
                 all_spn_x_y_model,
                 all_prior,
                 cnn=None,
                 get_max=True,
                 dtype=torch.float32,
                 gauss_embeds=0.01,
                 use_add_info=False,
                 VAE_fine_tune=0,
                 decoder=None,
                 loss_weights=None,
                 load_pretrain_model=1,
                 use_GAN=False,

                 filter_size=3,
                 num_layer=4,
                 input_shape=(128, 128, 1),
                 batchnorm_integration=1,
                 shortcut=0,
                 activation='relu',
                 num_filter=None,
                 strides=None,
                 end_dim=100,
                 dropout=0.2,
                 dilations=1,
                 discriminator_name='gan_discriminator',
                 clf_mlp=None

                 ):
        super(CNN_SPN_Parts, self).__init__()
        if num_filter is None:
            num_filter = [64, 128, 128]
        if strides is None:
            strides = [2, 2, 2, 2]
        if loss_weights is None:
            loss_weights = [1.0, 1.0, 1.0]

        self.clf_mlp = clf_mlp
        self.load_pretrain_model = load_pretrain_model
        self.all_spn_x_y = all_spn_x_y_model
        self.use_add_info = use_add_info
        self.decoder = decoder
        self.VAE_fine_tune = VAE_fine_tune
        self.loss_weights = loss_weights

        lw = loss_weights
        # Unwrap things like [[[10.0, 0.001, 5.0]]] → [10.0, 0.001, 5.0]
        while isinstance(lw, (list, tuple, np.ndarray)) and len(lw) == 1:
            lw = lw[0]

        # If after unwrapping it's not at least length 3, fall back to defaults
        if not (isinstance(lw, (list, tuple, np.ndarray)) and len(lw) >= 3):
            lw = [1.0, 1.0, 1.0]
        else:
            lw = [float(lw[0]), float(lw[1]), float(lw[2])]

        self.loss_weights = lw

        prior = torch.tensor(all_prior, dtype=dtype)
        self.register_buffer("prior_weights", prior)

        self.num_classes = num_classes
        self.get_max = get_max

        self.embedding = cnn

        self.spn_training = True
        self.clf = all_spn_x_y_model

        self.clf_loss = nn.NLLLoss()
        self.gauss_embed = gauss_embeds
        self.gauss_std = gauss_embeds

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.use_GAN = use_GAN
        if self.use_GAN:
            if activation == 'relu':
                act_fn = F.relu
            elif activation == 'tanh':
                act_fn = torch.tanh
            else:
                act_fn = F.relu

            self.discriminator = make_costume_encoder(
                filter_size, num_layer, input_shape,
                batchnorm_integration, num_filter, shortcut, strides,
                act_fn, discriminator_name, dtype, dilations, dropout, end_dim
            )

            self.gan_discr_optimizer = torch.optim.Adam(
                self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9)
            )
            self.gan_gener_optimizer = torch.optim.Adam(
                self.decoder.parameters(), lr=0.0001, betas=(0.5, 0.9)
            )

    def _maybe_gauss(self, embedding, training):
        if self.gauss_embed and training:
            noise = torch.randn_like(embedding) * self.gauss_std
            embedding = embedding + noise
        return embedding

    def model_execution_X_y(self, X, y):

        device = next(self.parameters()).device
        X = _to_tensor(X, dtype=torch.float32, device=device)
        y = _to_tensor(y, dtype=torch.long, device=device)

        out = self.embedding(X)
        if isinstance(out, tuple):
            embedding = out[0]
        else:
            embedding = out

        if self.get_max:
            embedding = torch.amax(embedding, dim=(1, 2))

        embedding = self._maybe_gauss(embedding, training=self.training)

        y_float = y.to(torch.float32).unsqueeze(-1)
        spn_input = torch.cat([y_float, embedding], dim=-1)

        weights = self.prior_weights.view(1, -1)
        inputs = []
        for sub_spn in self.all_spn_x_y:
            out = sub_spn(spn_input)
            if out.dim() > 1:
                out = out.squeeze(-1)
            inputs.append(out)
        children_prob = torch.stack(inputs, dim=1)  # [B, K]
        log_enumerator = children_prob + torch.log(weights)
        p_x = torch.logsumexp(log_enumerator, dim=1)
        p_y_x = log_enumerator - p_x.unsqueeze(1)
        return p_y_x

    def spn_clf(self, embedding, training):

        device = embedding.device
        embedding = embedding.to(device)

        weights = self.prior_weights.view(1, -1)  # [1, K]
        inputs = []
        for label_id, sub_spn in enumerate(self.all_spn_x_y):
            y = torch.full((embedding.size(0), 1), float(label_id),
                           device=device, dtype=embedding.dtype)
            spn_input = torch.cat([y, embedding], dim=-1)
            out = sub_spn(spn_input)
            if out.dim() > 1:
                out = out.squeeze(-1)
            inputs.append(out)
        children_prob = torch.stack(inputs, dim=1)  # [B, K]
        log_enumerator = children_prob + torch.log(weights)
        p_x = torch.logsumexp(log_enumerator, dim=1, keepdim=True)
        p_y_x = log_enumerator - p_x
        return p_y_x, p_x

    def model_execution_X(self, X, other_data, training=True):

        self.train(mode=training)
        device = next(self.parameters()).device
        X = _to_tensor(X, dtype=torch.float32, device=device)
        if other_data is not None and not isinstance(other_data, int):
            other_data = _to_tensor(other_data, dtype=torch.float32, device=device)
        else:
            other_data = None

        out = self.embedding(X)
        if isinstance(out, tuple):
            embedding = out[0]
        else:
            embedding = out

        if embedding.dim() > 2:
            embedding = embedding.view(embedding.size(0), -1)

        embedding = self._maybe_gauss(embedding, training=training)

        if self.use_add_info and other_data is not None:
            embedding = torch.cat([embedding, other_data], dim=-1)

        p_y_x, p_x = self.spn_clf(embedding, training)
        return p_y_x

    def train_step(self, x, y, other_data):

        self.train()
        device = next(self.parameters()).device
        x = _to_tensor(x, dtype=torch.float32, device=device)
        y = _to_tensor(y, dtype=torch.long, device=device)
        if other_data is not None and not isinstance(other_data, int):
            other_data = _to_tensor(other_data, dtype=torch.float32, device=device)
        else:
            other_data = None

        self.optimizer.zero_grad()
        spn_out = self.model_execution_X(x, other_data, training=True)
        loss = self.clf_loss(spn_out, y)
        loss.backward()
        self.optimizer.step()
        return [loss.detach()]

    def reconstruct(self, embedding):
        reconstruction = self.decoder(embedding)
        return reconstruction

    def vae_rec(self, X):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            X = _to_tensor(X, dtype=torch.float32, device=device)

            # Call embedding WITHOUT TF-style kwarg
            out = self.embedding(X)

            # Handle both (embedding,) tuple and plain tensor
            if isinstance(out, (tuple, list)):
                embedding = out[0]
            else:
                embedding = out

            return self.reconstruct(embedding)
        
    def model_execution_vae(self, X, y_real, other_data, training=True):

        self.train(mode=training)
        device = next(self.parameters()).device
        X = _to_tensor(X, dtype=torch.float32, device=device)
        y_real = _to_tensor(y_real, dtype=torch.long, device=device)
        if other_data is not None and not isinstance(other_data, int):
            other_data = _to_tensor(other_data, dtype=torch.float32, device=device)
        else:
            other_data = None

        # ---- normalize input if using pretrained model (keep your logic) ----
        normalized_X = X
        if self.load_pretrain_model:
            if normalized_X.dim() == 4 and normalized_X.shape[-1] > 1:
                normalized_X = normalized_X[..., 0:1]
            normalized_X = normalized_X / 255.0

        # ---- call embedding WITHOUT training kwarg, and handle shapes ----
        out = self.embedding(X)
        if isinstance(out, (tuple, list)):
            # if you ever make embedding return (z, mu, logvar)
            embedding_ = out[0]
            embed_mean = out[1] if len(out) > 1 else embedding_
            embed_var  = out[2] if len(out) > 2 else torch.zeros_like(embed_mean)
        else:
            # current case: clf_model() returns just mu
            embedding_ = out
            embed_mean = embedding_
            embed_var  = torch.zeros_like(embed_mean)

        # Decoder works on the latent embedding
        reconstruction = self.decoder(embedding_)

        # KL term; if embed_var is zeros this becomes ~0 (fine for fine-tuning)
        kl_loss = -0.5 * torch.sum(1 + embed_var - embed_mean.pow(2) - embed_var.exp(), dim=1)

        # ---- build embedding vector for SPN ----
        if self.get_max and embedding_.dim() > 2:
            # e.g. B x C x H x W → pool over spatial dims
            embedding = torch.amax(embedding_, dim=tuple(range(1, embedding_.dim())))
        else:
            # e.g. B x D → just flatten
            embedding = embedding_.view(embedding_.size(0), -1)

        embedding = self._maybe_gauss(embedding, training=training)

        if self.use_add_info and other_data is not None:
            embedding = torch.cat([embedding, other_data], dim=-1)

        # ---- SPN classification p(y|x,z) ----
        weights = self.prior_weights.view(1, -1)
        inputs = []
        for label_id, sub_spn in enumerate(self.all_spn_x_y):
            y = torch.full((embedding.size(0), 1), float(label_id),
                           device=device, dtype=embedding.dtype)
            spn_input = torch.cat([y, embedding], dim=-1)
            out_spn = sub_spn(spn_input)
            if out_spn.dim() > 1:
                out_spn = out_spn.squeeze(-1)
            inputs.append(out_spn)

        children_prob = torch.stack(inputs, dim=1)  # [B, K]
        log_enumerator = children_prob + torch.log(weights)
        p_x = torch.logsumexp(log_enumerator, dim=1, keepdim=True)
        p_y_x = log_enumerator - p_x

        clf_loss = self.clf_loss(p_y_x, y_real)

        # ---- reconstruction loss (match shapes, same logic as before) ----
        if reconstruction.shape != normalized_X.shape:
            if reconstruction.dim() == 4 and reconstruction.shape[1] == normalized_X.shape[-1]:
                reconstruction_for_loss = reconstruction.permute(0, 2, 3, 1)
            else:
                reconstruction_for_loss = reconstruction
        else:
            reconstruction_for_loss = reconstruction

        rec_loss_per_pixel = F.mse_loss(reconstruction_for_loss, normalized_X, reduction="none")
        while rec_loss_per_pixel.dim() > 1:
            rec_loss_per_pixel = rec_loss_per_pixel.mean(dim=-1)
        rec_loss = rec_loss_per_pixel

        # ---- combine losses ----
        rec_w, kl_w, clf_w = self.loss_weights
        loss = rec_loss * 2 * rec_w + kl_loss * kl_w + clf_loss * clf_w

        # average over batch
        rec_loss = rec_loss.mean()
        kl_loss = kl_loss.mean()
        loss = loss.mean()

        return p_y_x, loss, rec_loss, clf_loss, kl_loss, embedding_


    def model_execution_vae_eval(self, X, y_real, other_data):
        training = False
        self.eval()
        device = next(self.parameters()).device
        X = _to_tensor(X, dtype=torch.float32, device=device)
        y_real = _to_tensor(y_real, dtype=torch.long, device=device)
        if other_data is not None and not isinstance(other_data, int):
            other_data = _to_tensor(other_data, dtype=torch.float32, device=device)
        else:
            other_data = None

        normalized_X = X
        if self.load_pretrain_model:
            if normalized_X.dim() == 4 and normalized_X.shape[-1] > 1:
                normalized_X = normalized_X[..., 0:1]
            normalized_X = normalized_X / 255.0

        with torch.no_grad():
            out = self.embedding(X)
            if isinstance(out, (tuple, list)):
                embedding_ = out[0]
                embed_mean = out[1] if len(out) > 1 else embedding_
                embed_var  = out[2] if len(out) > 2 else torch.zeros_like(embed_mean)
            else:
                embedding_ = out
                embed_mean = embedding_
                embed_var  = torch.zeros_like(embed_mean)

            reconstruction = self.decoder(embedding_)

            kl_loss = -0.5 * torch.sum(1 + embed_var - embed_mean.pow(2) - embed_var.exp(), dim=1)

            if self.get_max and embedding_.dim() > 2:
                embedding = torch.amax(embedding_, dim=tuple(range(1, embedding_.dim())))
            else:
                embedding = embedding_.view(embedding_.size(0), -1)

            embedding = self._maybe_gauss(embedding, training=training)

            if self.use_add_info and other_data is not None:
                embedding = torch.cat([embedding, other_data], dim=-1)

            weights = self.prior_weights.view(1, -1)
            inputs = []
            for label_id, sub_spn in enumerate(self.all_spn_x_y):
                y = torch.full((embedding.size(0), 1), float(label_id),
                               device=device, dtype=embedding.dtype)
                spn_input = torch.cat([y, embedding], dim=-1)
                out_spn = sub_spn(spn_input)
                if out_spn.dim() > 1:
                    out_spn = out_spn.squeeze(-1)
                inputs.append(out_spn)
            children_prob = torch.stack(inputs, dim=1)  # [B, K]
            log_enumerator = children_prob + torch.log(weights)
            p_x = torch.logsumexp(log_enumerator, dim=1, keepdim=True)
            p_y_x = log_enumerator - p_x

            if self.clf_mlp is not None:
                emb_mlp = embedding_
                # If it's not already (B, D), flatten it
                if emb_mlp.dim() > 2:
                    emb_mlp = emb_mlp.view(emb_mlp.size(0), -1)
                p_y_x_mlp = self.clf_mlp(emb_mlp)
            else:
                p_y_x_mlp = None

            if reconstruction.shape != normalized_X.shape:
                if reconstruction.dim() == 4 and reconstruction.shape[1] == normalized_X.shape[-1]:
                    reconstruction_for_loss = reconstruction.permute(0, 2, 3, 1)
                else:
                    reconstruction_for_loss = reconstruction
            else:
                reconstruction_for_loss = reconstruction

            clf_loss = self.clf_loss(p_y_x, y_real)

            rec_loss_per_pixel = F.mse_loss(reconstruction_for_loss, normalized_X, reduction="none")
            while rec_loss_per_pixel.dim() > 1:
                rec_loss_per_pixel = rec_loss_per_pixel.mean(dim=-1)
            rec_loss = rec_loss_per_pixel

            mae_per_pixel = F.l1_loss(reconstruction_for_loss, normalized_X, reduction="none")
            while mae_per_pixel.dim() > 1:
                mae_per_pixel = mae_per_pixel.mean(dim=-1)
            mae = mae_per_pixel

            rec_w, kl_w, clf_w = self.loss_weights
            loss = rec_loss * 2 * rec_w + kl_loss * kl_w + clf_loss * clf_w

            rec_loss = rec_loss.mean()
            kl_loss = kl_loss.mean()
            loss = loss.mean()
            mae = mae.mean()

        return p_y_x, loss, rec_loss, clf_loss, kl_loss, mae, embedding_, p_y_x_mlp


    def train_step_vae_one_loss(self, x, y, other_data):

        self.train()
        device = next(self.parameters()).device
        x = _to_tensor(x, dtype=torch.float32, device=device)
        y = _to_tensor(y, dtype=torch.long, device=device)
        if other_data is not None and not isinstance(other_data, int):
            other_data = _to_tensor(other_data, dtype=torch.float32, device=device)
        else:
            other_data = None

        self.optimizer.zero_grad()
        spn_out, loss, rec_loss, clf_loss, kl_loss, z = self.model_execution_vae(x, y, other_data, training=True)
        loss.backward()
        self.optimizer.step()
        return [loss.detach(), rec_loss.detach(), clf_loss.detach(), kl_loss.detach()], z.detach(), spn_out.detach()

    def gan_step(self, X, y, other_info, discriminator_loss_old, generator_loss_old):

        device = next(self.parameters()).device
        X = _to_tensor(X, dtype=torch.float32, device=device)

        for _ in range(5):
            self.gan_discr_optimizer.zero_grad()
            fake_images = self.decoder(self.embedding(X)[0])
            real_images = X[..., 0:1] / 255.0
            fake_out = self.discriminator(torch.sigmoid(fake_images))
            real_out = self.discriminator(real_images)

            gp = gradient_penalty(self.discriminator, real_images, torch.sigmoid(fake_images))
            gan_loss = fake_out.mean() - real_out.mean() + 10 * gp
            gan_loss.backward()
            self.gan_discr_optimizer.step()

        self.gan_gener_optimizer.zero_grad()
        fake_images = self.decoder(self.embedding(X)[0])
        fake_out = self.discriminator(torch.sigmoid(fake_images))
        generator_loss = -fake_out.mean()
        generator_loss.backward()
        self.gan_gener_optimizer.step()

        return gan_loss.detach(), generator_loss.detach()

    def train_gan(self, train_ds, first_loss):
        counter = 0.0
        all_losses = np.zeros(6, dtype=np.float64)
        z_train, gt_train, predictions_train = [], [], []
        discriminator_loss_old, generator_loss_old = 0, 0
        for it_c, train_rec in enumerate(train_ds):
            other_info = 0
            if self.use_add_info:
                (X, y, other_info) = train_rec
            else:
                (X, y) = train_rec

            loss, z, pred = self.train_step_vae_one_loss(X, y, other_info)
            discriminator_loss, generator_loss = self.gan_step(
                X, y, other_info, discriminator_loss_old, generator_loss_old
            )

            z_train.append(z.cpu().numpy())
            gt_train.append(y)
            predictions_train.append(pred.cpu().numpy())

            loss_vals = [entry.cpu().item() for entry in loss]
            loss_vals.append(discriminator_loss.cpu().item())
            loss_vals.append(generator_loss.cpu().item())
            discriminator_loss_old, generator_loss_old = discriminator_loss.cpu().item(), generator_loss.cpu().item()

            all_losses += np.array(loss_vals, dtype=np.float64)
            counter += 1
            if not it_c and first_loss:
                print(it_c, 'loss', all_losses / counter)
        return [all_losses / counter, z_train, gt_train, predictions_train]

    def train_step_vae_diff_loss(self, x, y, other_data):
        self.train()
        device = next(self.parameters()).device
        x = _to_tensor(x, dtype=torch.float32, device=device)
        y = _to_tensor(y, dtype=torch.long, device=device)
        if other_data is not None and not isinstance(other_data, int):
            other_data = _to_tensor(other_data, dtype=torch.float32, device=device)
        else:
            other_data = None

        self.optimizer.zero_grad()
        spn_out = self.model_execution_X(x, other_data, training=True)
        loss = self.clf_loss(spn_out, y)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def train_model(self, train_ds, first_loss):
        """
        self.train(...)
        """
        counter = 0.0
        if self.VAE_fine_tune == 0:
            all_losses = np.zeros(1, dtype=np.float64)
            for it_c, train_rec in enumerate(train_ds):
                other_info = 0
                if self.use_add_info:
                    (X, y, other_info) = train_rec
                else:
                    (X, y) = train_rec

                loss_list = self.train_step(X, y, other_info)
                new_loss = [entry.cpu().item() for entry in loss_list]
                all_losses += np.array(new_loss, dtype=np.float64)
                counter += 1
                if not it_c and first_loss:
                    print(it_c, 'loss', all_losses / counter)
            return all_losses / counter
        elif self.VAE_fine_tune == 1:
            all_losses = np.zeros(4, dtype=np.float64)
            z_train, gt_train, predictions_train = [], [], []
            for it_c, train_rec in enumerate(train_ds):
                other_info = 0
                if self.use_add_info:
                    (X, y, other_info) = train_rec
                else:
                    (X, y) = train_rec

                loss_list, z, pred = self.train_step_vae_one_loss(X, y, other_info)

                z_train.append(z.cpu().numpy())
                gt_train.append(y)
                predictions_train.append(pred.cpu().numpy())

                new_loss = [entry.cpu().item() for entry in loss_list]
                all_losses += np.array(new_loss, dtype=np.float64)
                counter += 1
                if not it_c and first_loss:
                    print(it_c, 'loss', all_losses / counter)
            return [all_losses / counter, z_train, gt_train, predictions_train]

        elif self.VAE_fine_tune == 2:
            all_losses = np.zeros(4, dtype=np.float64)
            for it_c, train_rec in enumerate(train_ds):
                other_info = 0
                if self.use_add_info:
                    (X, y, other_info) = train_rec
                else:
                    (X, y) = train_rec

                loss = self.train_step_vae_diff_loss(X, y, other_info)
                new_loss = [loss.cpu().item()]
                all_losses += np.array(new_loss, dtype=np.float64)
                counter += 1
                if not it_c and first_loss:
                    print(it_c, 'loss', all_losses / counter)
            return all_losses / counter

    def eval_cnn(self, test_X):
        self.eval()
        with torch.no_grad():
            x = _to_tensor(test_X, dtype=torch.float32, device=next(self.parameters()).device)
            return self.embedding(x)

    def get_spn_variables(self):
        vars_list = []
        for sub_spn in self.all_spn_x_y:
            if hasattr(sub_spn, "parameters"):
                vars_list.extend(list(sub_spn.parameters()))
        return vars_list


def train_model_parts(grid_params, cnn_spn, train_data, val_data, test_data,
                      num_iterations, ckpt_path=None, manager=None, val_entropy=0, val_acc=0,
                      add_info=False):


    device = next(cnn_spn.parameters()).device

    first_loss = True

    num_params = sum(p.numel() for p in cnn_spn.parameters() if p.requires_grad)
    print('number of trainable variables in cnn spn:', num_params)

    train_start_time = time.time()
    best_acc_loss = val_acc
    best_val_reconstruction = 1e8
    eval_after_train = []
    all_debugging_stuff = []

    for i in range(num_iterations):
        idx_train = list(range(train_data[0].shape[0]))
        random.shuffle(idx_train)

        X = train_data[0][idx_train]

        if add_info:
            y = train_data[1][idx_train, 0]
            other = train_data[1][idx_train, 1:]

            X_t = torch.from_numpy(X).to(device=device, dtype=torch.float32)
            y_t = torch.from_numpy(y).to(device=device, dtype=torch.long)
            other_t = torch.from_numpy(other).to(device=device, dtype=torch.float32)

            if getattr(grid_params, "use_add_info", False):
                dataset = TensorDataset(X_t, y_t, other_t)
            else:
                dataset = TensorDataset(X_t, y_t)
        else:
            y = train_data[1][idx_train]
            X_t = torch.from_numpy(X).to(device=device, dtype=torch.float32)
            y_t = torch.from_numpy(y).to(device=device, dtype=torch.long)
            dataset = TensorDataset(X_t, y_t)

        train_loader = DataLoader(
            dataset,
            batch_size=grid_params.batch_size,
            shuffle=False
        )

        if grid_params.VAE_fine_tune and not getattr(grid_params, "GAN", False):

            train_loss, z_train, gt_train, predictions_train = cnn_spn.train_model(
                train_loader, first_loss=first_loss
            )
        elif grid_params.VAE_fine_tune and getattr(grid_params, "GAN", False):

            train_loss, z_train, gt_train, predictions_train = cnn_spn.train_gan(
                train_loader, first_loss=first_loss
            )
        else:

            train_loss = cnn_spn.train_model(train_loader, first_loss=first_loss)

        first_loss = False

        if i % 3 == 0:
            improve = False

            if grid_params.VAE_fine_tune:

                val_losses, z, gt, arg_max, pred_exp = test_model_all(
                    cnn_spn,
                    val_data,
                    num_classes=2,
                    batch_size=grid_params.batch_size,
                    training=False,
                    add_info=add_info
                )
                curr_acc = val_losses[2]

                if curr_acc > best_acc_loss:
                    best_acc_loss = curr_acc
                    improve = True

                    eval_after_train = test_model_no_mpe(
                        cnn_spn,
                        [train_data, val_data, test_data],
                        cnn_spn.num_classes,
                        grid_params.batch_size,
                        add_info=add_info
                    )

            else:
                [val_losses] = test_model_no_mpe(
                    cnn_spn,
                    [val_data],
                    num_classes=2,
                    batch_size=grid_params.batch_size,
                    training=False,
                    add_info=add_info
                )
                curr_acc = val_losses[2]

                if curr_acc > best_acc_loss:
                    best_acc_loss = curr_acc
                    improve = True

                    eval_after_train = test_model_no_mpe(
                        cnn_spn,
                        [train_data, val_data, test_data],
                        cnn_spn.num_classes,
                        grid_params.batch_size,
                        add_info=add_info
                    )

            if improve and ckpt_path is not None:
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(cnn_spn.state_dict(), ckpt_path)
                print(f"Saved CNN+SPN checkpoint to {ckpt_path}")

            print(
                'val entropy', i,
                val_losses[1],
                'improve', improve,
                'curr acc', val_losses[0], val_losses[2],
                'best acc', best_acc_loss
            )

            if grid_params.VAE_fine_tune:
                print('curr rec N/A', 'best rec:', best_val_reconstruction)

        print('loss', i, end=': ')
        if isinstance(train_loss, (np.ndarray, list, tuple)):
            for loss_val in train_loss:
                loss_val = float(loss_val)
                print(np.round(loss_val, 5), end=' - ')
        else:
            print(np.round(float(train_loss), 5), end=' - ')
        print()

    print('fine tune train time', (time.time() - train_start_time) // 60)
    return eval_after_train, all_debugging_stuff


def test_model_no_mpe(tfmodel, data_sets, num_classes, batch_size=25, training=False, add_info=False):
    device = next(tfmodel.parameters()).device
    all_evals = []
    for dataset in data_sets:
        prediction = []
        gt = []
        for i in range(dataset[0].shape[0] // batch_size):
            X_data = dataset[0][i * batch_size:(i + 1) * batch_size]
            other_data = 0
            if add_info:
                y_data = dataset[1][i * batch_size:(i + 1) * batch_size, 0]
                other_data = dataset[1][i * batch_size:(i + 1) * batch_size, 1:]
            else:
                y_data = dataset[1][i * batch_size:(i + 1) * batch_size]

            X_t = torch.from_numpy(X_data).to(device=device, dtype=torch.float32)
            if isinstance(y_data, np.ndarray):
                y_t = torch.from_numpy(y_data).to(device=device, dtype=torch.long)
            else:
                y_t = torch.tensor(y_data, device=device, dtype=torch.long)

            if add_info and isinstance(other_data, np.ndarray):
                other_t = torch.from_numpy(other_data).to(device=device, dtype=torch.float32)
            else:
                other_t = None

            pred = tfmodel.model_execution_X(X_t, other_t, training=False)
            prediction.extend(pred.detach().cpu().numpy().tolist())
            gt.extend(y_t.detach().cpu().numpy().tolist())

        prediction = np.asarray(prediction)
        gt_np = np.asarray(gt)

        logits_t = torch.from_numpy(prediction).to(device=device, dtype=torch.float32)
        gt_t = torch.from_numpy(gt_np).to(device=device, dtype=torch.long)
        entropy = tfmodel.clf_loss(logits_t, gt_t).item()

        prediction_exponential = np.exp(prediction)
        arg_max = np.argmax(prediction_exponential, axis=-1)
        acc = accuracy_score(gt_np, arg_max, normalize=True)

        auc = 0
        if num_classes == 2:
            pred_exp = np.nan_to_num(
                prediction_exponential[:, 1],
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            fpr, tpr, thresholds = metrics.roc_curve(gt_np, pred_exp, pos_label=1)
            auc = metrics.auc(fpr, tpr)
        balanced_acc = balanced_accuracy_score(gt_np, arg_max)
        [prec, rec, f1, _] = precision_recall_fscore_support(gt_np, arg_max, average=None)
        results = [acc, entropy, balanced_acc, prec[1], rec[1], f1[1], auc]
        all_evals.append(results)
        print(results)
    return all_evals


def test_model_SPN_MLP(tfmodel, dataset, num_classes, batch_size=25, training=False, add_info=False):
    mlp_prediction = []
    prediction = []
    gt = []
    z = []
    losses = np.zeros(5, dtype=np.float64)
    n_samples = dataset[0].shape[0]
    for i in range(n_samples // batch_size):
        X_data = dataset[0][i * batch_size:(i + 1) * batch_size]
        other_data = 0
        if add_info:
            y_data = dataset[1][i * batch_size:(i + 1) * batch_size, 0]
            other_data = dataset[1][i * batch_size:(i + 1) * batch_size, 1:]
        else:
            y_data = dataset[1][i * batch_size:(i + 1) * batch_size]

        pred, loss, rec_loss, clf_loss_val, kl_loss, mae, embedding_, mlp_pred = tfmodel.model_execution_vae_eval(
            X_data, y_data, other_data
        )
        new_loss = np.asarray(
            [loss.item(), rec_loss.item(), clf_loss_val.item(), kl_loss.item(), mae.item()],
            dtype=np.float64
        )
        losses += new_loss
        prediction.extend(pred.detach().cpu().numpy().tolist())
        mlp_prediction.extend(mlp_pred.detach().cpu().numpy().tolist())
        gt.extend(y_data.tolist())
        z.extend(embedding_.detach().cpu().numpy().tolist())
    losses /= n_samples // batch_size
    prediction = np.asarray(prediction)
    mlp_prediction = np.asarray(mlp_prediction)

    prediction_exponential = np.exp(prediction)

    results_MLP, _, _ = eval_cls(mlp_prediction, mlp_prediction, gt, tfmodel.clf_loss, num_classes)
    results_SPN, _, _ = eval_cls(prediction, prediction_exponential, gt, tfmodel.clf_loss, num_classes)

    return results_MLP, results_SPN, losses


def test_model_all(tfmodel, dataset, num_classes, batch_size=25, training=False, add_info=False):
    prediction = []
    gt = []
    z = []
    losses = np.zeros(4, dtype=np.float64)
    n_samples = dataset[0].shape[0]
    for i in range(n_samples // batch_size):
        X_data = dataset[0][i * batch_size:(i + 1) * batch_size]
        other_data = 0
        if add_info:
            y_data = dataset[1][i * batch_size:(i + 1) * batch_size, 0]
            other_data = dataset[1][i * batch_size:(i + 1) * batch_size, 1:]
        else:
            y_data = dataset[1][i * batch_size:(i + 1) * batch_size]

        pred, loss, rec_loss, clf_loss_val, kl_loss, mae, embedding_, mlp_pred = tfmodel.model_execution_vae_eval(
            X_data, y_data, other_data
        )
        new_loss = np.asarray(
            [loss.item(), rec_loss.item(), clf_loss_val.item(), kl_loss.item()],
            dtype=np.float64
        )
        losses += new_loss
        prediction.extend(pred.detach().cpu().numpy().tolist())
        gt.extend(y_data.tolist())
        z.extend(embedding_.detach().cpu().numpy().tolist())
    losses /= n_samples // batch_size
    prediction = np.asarray(prediction)

    prediction_exponential = np.exp(prediction)

    results, pred_exp, arg_max = eval_cls(
        prediction, prediction_exponential, gt, tfmodel.clf_loss, num_classes
    )
    return results, z, gt, arg_max, pred_exp


def eval_cls(pred_logits, prediction_exponential, gt, clf_loss, num_classes=2):
    device = torch.device("cpu")
    pred_arg_max = np.argmax(prediction_exponential, axis=-1)

    logits_t = torch.from_numpy(pred_logits).to(dtype=torch.float32, device=device)
    gt_t = torch.tensor(gt, dtype=torch.long, device=device)
    entropy = clf_loss(logits_t, gt_t).item()

    acc = accuracy_score(gt, pred_arg_max, normalize=True)

    auc = 0
    pred_exp = None
    if num_classes == 2:
        pred_exp = np.nan_to_num(prediction_exponential[:, 1], nan=0, posinf=1.0)
        fpr, tpr, thresholds = metrics.roc_curve(gt, pred_exp, pos_label=1)
        auc = metrics.auc(fpr, tpr)
    balanced_acc = balanced_accuracy_score(gt, pred_arg_max)
    prec, rec, f1, _ = precision_recall_fscore_support(gt, pred_arg_max, average=None)

    return [acc, entropy, balanced_acc, prec[1], rec[1], f1[1], auc], pred_exp, pred_arg_max