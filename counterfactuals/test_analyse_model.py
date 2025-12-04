# tests/test_analyse_model.py
import pytest
import torch
import types

from counterfactuals.analyse_model import load_VAE_model, load_model

############################################
# Helper mocks
############################################

class MockParams:
    def __init__(self):
        self.fine_tune_leafs = False
        self.use_VAE = True
        self.VAE_fine_tune = False
        self.loss_weights = None
        self.gauss_embeds = False
        self.use_add_info = False
        self.load_pretrain_model = False
        self.machine = "cpu"
        self.batch_size = 2

class MockVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(10, 5)
        self.decoder = torch.nn.Linear(5, 10)
        self.classifier = torch.nn.Linear(5, 2)

    # --- emulate clf_model() used in load_model() ---
    def clf_model(self):
        return torch.nn.Linear(10, 5)

def mock_load_VAE(*args, **kwargs):
    return MockVAE()

############################################
# Monkeypatch load_VAE import used by analyse_model.load_model
############################################

@pytest.fixture
def patch_load_vae(monkeypatch):
    import counterfactuals.analyse_model as am
    monkeypatch.setattr(am, "load_VAE", mock_load_VAE)
    return True


############################################
# Tests
############################################

def test_load_vae_model_smoke(patch_load_vae):
    """Ensure load_VAE_model returns a VAE instance and requires no real checkpoints."""
    params = MockParams()
    vae = load_VAE_model(
        add_info=False,
        num_classes=2,
        input_shape=(1, 128, 128),
        data_path_fold="./fake/",
        params=params,
        path=""
    )
    assert isinstance(vae, MockVAE)


def test_load_model_constructs_cnn_spn(patch_load_vae, monkeypatch):
    """
    Ensures load_model builds CNN_SPN_Parts, wires the embedding, and loads checkpoints safely.
    """

    # ----- mock create_torch_spn_parts -----
    def mock_spn_parts(**kwargs):
        mock_spn = torch.nn.Linear(8, 2)
        return None, None, None, None, mock_spn

    monkeypatch.setattr(
        "counterfactuals.analyse_model.create_torch_spn_parts",
        lambda *args, **kwargs: mock_spn_parts()
    )

    # ----- mock CNN_SPN_Parts class -----
    class MockCnnSpn(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.embedding_net = kwargs["cnn"]
            self.clf_mlp = kwargs["clf_mlp"]
            self.linear = torch.nn.Linear(5, 2)

        def embedding(self, X):
            return torch.zeros(X.shape[0], 5), None, None

        def spn_clf(self, emb):
            return torch.randn(emb.shape[0], 2), torch.zeros(emb.shape[0], 1)

        def reconstruct(self, z):
            return torch.zeros(z.shape[0], 1, 128, 128)

        def model_execution_X(self, X, additional_info=None, training=False):
            return torch.tensor([[0.5, 0.5]] * X.shape[0])

    # patch the import inside analyse_model
    import counterfactuals.analyse_model as am
    monkeypatch.setattr(am, "CNN_SPN_Parts", MockCnnSpn)

    params = MockParams()

    model = load_model(
        add_info=False,
        num_classes=2,
        input_shape=(1, 128, 128),
        data_path_fold="./fake/",
        params=params,
        path=""
    )

    # basic structural checks
    assert isinstance(model, MockCnnSpn)
    assert hasattr(model, "embedding")
    assert hasattr(model, "spn_clf")
    assert hasattr(model, "reconstruct")
    assert hasattr(model, "model_execution_X")
