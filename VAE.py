import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=62):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU(inplace=True)
        )
        self.fc_mu = nn.Linear(512 * 1 * 1, latent_dim)
        self.fc_logvar = nn.Linear(512 * 1 * 1, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_channels=1, latent_dim=62):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, out_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 512, 1, 1)
        x_recon = self.deconv(h)
        return x_recon


class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=62, img_size=128, num_classes = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

        # classifier head to match TF pipeline - dennis
        self.classifier = nn.Linear(latent_dim, num_classes)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        logits = self.classifier(mu)
        return x_recon, mu, logvar, logits
    def clf_model(self):
        # provide latent embedding extraction
        class LatentExtractor(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.encoder = parent.encoder
            def forward(self, x):
                if x.dim() == 4 and x.shape[1] not in (1, 3):
                    x = x.permute(0, 3, 1, 2).contiguous()
                mu, logvar = self.encoder(x)
                return mu
        return LatentExtractor(self)
    def vae_rec(self, x):
        # reconstruction-only method. I don't think this'll be used since it's in CNN_SPN but just in case.
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x).float()
            if x.ndim == 3:
                x = x.unsqueeze(0)
            recon, _, _, = self.forward(x)
        return recon.cpu().numpy()

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


def train_vae(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_samples = 0
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(device).float()
        x = to_nchw(x)  # <<< important: NHWC -> NCHW if needed

        optimizer.zero_grad()
        recon_x, mu, logvar, logits = model(x) #RASHED ADDED 4th OUTPUT (logits) BUT NOT USING IT HERE for unpacking error
        loss = vae_loss(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        batch_size = x.size(0)
        total_loss += loss.item()
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def test_vae(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(device).float()
            x = to_nchw(x)  # <<< same permutation

            recon_x, mu, logvar, logits = model(x) #RASHED ADDED 4th OUTPUT (logits) BUT NOT USING IT HERE for unpacking error
            loss = vae_loss(recon_x, x, mu, logvar) 
            batch_size = x.size(0)
            total_loss += loss.item()
            total_samples += batch_size

    return total_loss / max(total_samples, 1)

def to_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert from NHWC (B, H, W, C) to NCHW (B, C, H, W) if needed.
    If already NCHW (channels in dim 1), returns as-is.
    """
    if x.dim() == 4 and x.shape[1] not in (1, 3):
        # Assume NHWC and permute to NCHW
        x = x.permute(0, 3, 1, 2).contiguous()
    return x