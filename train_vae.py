import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from VAE import VAE, train_vae, test_vae

class ChexpertDataset(Dataset):
    def __init__(self, split):
        with open("chexpert.pkl", "rb") as f:
            data = pickle.load(f)
        imgs = data[split][0]
        self.images = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1) / 255.0  # (N,1,128,128)
        self.labels = torch.tensor(data[split][1], dtype=torch.float32)  # 可忽略
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = ChexpertDataset("train")
    val_set = ChexpertDataset("validation")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    model = VAE(in_channels=1, latent_dim=62).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, 11):
        loss = train_vae(model, train_loader, optimizer, device)
        val_loss = test_vae(model, val_loader, device)
        print(f"Epoch {epoch}: Train Loss={loss:.4f}, Val Loss={val_loss:.4f}")

    torch.save(model.state_dict(), "vae_trained.pt")

if __name__ == "__main__":
    main()
