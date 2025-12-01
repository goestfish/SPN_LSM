import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace

from torch_CNN_SPN import CNN_SPN, CNN_SPN_Parts, train_model_parts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_toy_batch_linear(batch_size=64, input_dim=32, seed=0):

    rng = np.random.RandomState(seed)
    X_np = rng.randn(batch_size, input_dim).astype("float32")
    y_np = (X_np.sum(axis=1) > 0).astype("int64")
    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    return X, y


def make_toy_images(
    N=128, H=8, W=8, C=1, num_classes=2, seed=0
):
    rng = np.random.RandomState(seed)
    X = rng.randn(N, H, W, C).astype("float32")

    y = (X.sum(axis=(1, 2, 3)) > 0).astype("int64")
    return X, y

class DummyEncoder(nn.Module):
    def __init__(self, input_flat_dim, z_dim=16):
        super().__init__()
        self.fc_mu = nn.Linear(input_flat_dim, z_dim)
        self.fc_logvar = nn.Linear(input_flat_dim, z_dim)

    def forward(self, x, training=True):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class DummyDecoder(nn.Module):
    def __init__(self, z_dim, out_shape):
        super().__init__()
        self.out_shape = out_shape
        out_flat = 1
        for d in out_shape:
            out_flat *= d
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_flat),
        )

    def forward(self, z):
        x_hat = self.net(z)
        return x_hat.view(z.size(0), *self.out_shape)


class DummySPN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x, training=True):
        return self.net(x)

class DummyCkpt:
    def __init__(self):
        self.step = SimpleNamespace(assign_add=lambda x: None)


class DummyManager:
    def __init__(self, name=""):
        self.name = name

    def save(self):
        pass

def test_cnn_spn_basic():
    print("========== [Test 1] CNN_SPN basic ==========")
    input_dim = 32
    num_classes = 2
    lr = 1e-2

    model = CNN_SPN(
        num_classes=num_classes,
        input_dimensions=(input_dim,),
        learning_rate=lr,
        spn=None,
        cnn=None,
        get_max=False,
    ).to(device)

    if not hasattr(model, "use_add_info"):
        model.use_add_info = False

    X, y = make_toy_batch_linear(batch_size=64, input_dim=input_dim, seed=123)

    with torch.no_grad():
        loss = model.model_execution_X_y(X, y, other=None)
    print(f"[CNN_SPN] forward loss: {float(loss):.6f}")

    print("[CNN_SPN] start small train loop...")
    for step in range(20):
        loss = model.train_step(X, y, other=None)
        print(f"  step {step:02d}, loss = {float(loss):.6f}")
    print()

def test_cnn_spn_parts_forward():
    print("========== [Test 2] CNN_SPN_Parts forward ==========")
    H, W, C = 8, 8, 1
    num_classes = 2
    lr = 1e-3
    z_dim = 16

    X_np, y_np = make_toy_images(N=16, H=H, W=W, C=C, num_classes=num_classes, seed=0)

    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)

    input_flat_dim = H * W * C
    encoder = DummyEncoder(input_flat_dim=input_flat_dim, z_dim=z_dim)
    decoder = DummyDecoder(z_dim=z_dim, out_shape=(H, W, C))
    spn_in_dim = 1 + z_dim
    all_spn_x_y = [DummySPN(spn_in_dim) for _ in range(num_classes)]
    priors = [1.0 / num_classes] * num_classes

    model = CNN_SPN_Parts(
        num_classes=num_classes,
        learning_rate=lr,
        all_spn_x_y_model=all_spn_x_y,
        all_prior=priors,
        cnn=encoder,
        get_max=False,
        gauss_embeds=0.0,
        use_add_info=False,
        VAE_fine_tune=0,
        decoder=decoder,
        loss_weights=[1.0, 1.0, 1.0],
        load_pretrain_model=0,
        use_GAN=False,
        filter_size=3,
        num_layer=4,
        input_shape=(H, W, C),
        batchnorm_integration=1,
        shortcut=0,
        activation="relu",
        num_filter=[64, 128, 128],
        strides=[2, 2, 2, 2],
        end_dim=100,
        dropout=0.2,
        dilations=1,
        discriminator_name="gan_discriminator",
        clf_mlp=None,
    ).to(device)

    if not hasattr(model, "use_add_info"):
        model.use_add_info = False

    with torch.no_grad():
        out = model.model_execution_X_y(X, y)
    print("[CNN_SPN_Parts] model_execution_X_y output shape:", out.shape)

    with torch.no_grad():
        logits = model.model_execution_X(X, other_data=None, training=False)
    print("[CNN_SPN_Parts] model_execution_X output shape:", logits.shape)

    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    print("[CNN_SPN_Parts] probs[0]:", probs[0].cpu().numpy())
    print("[CNN_SPN_Parts] preds:", preds.cpu().numpy())
    print()


def test_train_model_parts_loop():
    print("========== [Test 3] train_model_parts loop ==========")

    H, W, C = 8, 8, 1
    num_classes = 2
    z_dim = 16

    X_train, y_train = make_toy_images(N=128, H=H, W=W, C=C, num_classes=num_classes, seed=1)
    X_val,   y_val   = make_toy_images(N=64,  H=H, W=W, C=C, num_classes=num_classes, seed=2)
    X_test,  y_test  = make_toy_images(N=64,  H=H, W=W, C=C, num_classes=num_classes, seed=3)

    train_data = (X_train, y_train)
    val_data   = (X_val, y_val)
    test_data  = (X_test, y_test)

    input_flat_dim = H * W * C
    encoder = DummyEncoder(input_flat_dim=input_flat_dim, z_dim=z_dim)
    decoder = DummyDecoder(z_dim=z_dim, out_shape=(H, W, C))
    spn_in_dim = 1 + z_dim
    all_spn_x_y = [DummySPN(spn_in_dim) for _ in range(num_classes)]
    priors = [1.0 / num_classes] * num_classes

    cnn_spn = CNN_SPN_Parts(
        num_classes=num_classes,
        learning_rate=1e-3,
        all_spn_x_y_model=all_spn_x_y,
        all_prior=priors,
        cnn=encoder,
        get_max=False,
        gauss_embeds=0.0,
        use_add_info=False,
        VAE_fine_tune=0,
        decoder=decoder,
        loss_weights=[1.0, 1.0, 1.0],
        load_pretrain_model=0,
        use_GAN=False,
        filter_size=3,
        num_layer=4,
        input_shape=(H, W, C),
        batchnorm_integration=1,
        shortcut=0,
        activation="relu",
        num_filter=[64, 128, 128],
        strides=[2, 2, 2, 2],
        end_dim=100,
        dropout=0.2,
        dilations=1,
        discriminator_name="gan_discriminator",
        clf_mlp=None,
    ).to(device)

    if not hasattr(cnn_spn, "use_add_info"):
        cnn_spn.use_add_info = False

    grid_params = SimpleNamespace(
        use_add_info=False,
        batch_size=16,
        VAE_fine_tune=0,
        GAN=False,
    )

    ckpt = [DummyCkpt(), DummyCkpt()]
    manager = [DummyManager("vae"), DummyManager("spn")]

    val_entropy_init = 0.0

    eval_after_train, debug_info = train_model_parts(
        grid_params=grid_params,
        cnn_spn=cnn_spn,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        num_iterations=5,
        ckpt=ckpt,
        manager=manager,
        val_entropy=val_entropy_init,
        val_acc=0.0,
        add_info=False,
    )

    print("[train_model_parts] eval_after_train:", eval_after_train)
    print("[train_model_parts] len(debug_info):", len(debug_info))
    print()


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    test_cnn_spn_basic()

    test_cnn_spn_parts_forward()

    test_train_model_parts_loop()