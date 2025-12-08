import re
import matplotlib.pyplot as plt

log_path = "fold0results.txt"

train_losses = []
val_losses   = []

with open(log_path, "r") as f:
    for line in f:
        match = re.search(
            r"Epoch\s+(\d+)\s+train_loss=([0-9.]+)\s+val_loss=([0-9.]+)", 
            line
        )
        if match:
            epoch = int(match.group(1))
            train_losses.append(float(match.group(2)))
            val_losses.append(float(match.group(3)))

plt.figure(figsize=(7,4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Training Curve (Fold 0)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vae_training_curve_fold0.png", dpi=300, bbox_inches='tight')
plt.show()


