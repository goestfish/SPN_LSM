import numpy as np
import torch

from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.structure.Base import Sum, Product

from spn.algorithms.Inference import log_likelihood

from torch_spn import create_torch_spn


def build_toy_spn():

    g00 = Gaussian(mean=0.0, stdev=1.0, scope=0)
    g01 = Gaussian(mean=3.0, stdev=1.0, scope=0)

    g10 = Gaussian(mean=0.0, stdev=1.0, scope=1)
    g11 = Gaussian(mean=-3.0, stdev=1.0, scope=1)

    p0 = Product(children=[g00, g10])
    p1 = Product(children=[g01, g11])

    root = Sum(weights=[0.4, 0.6], children=[p0, p1])

    return root


def main():
    spn_root = build_toy_spn()

    torch_spn_model, var_dict, spn_copy = create_torch_spn(spn_root)

    B = 5
    x_np = np.random.randn(B, 2).astype(np.float32)
    x_torch = torch.from_numpy(x_np)

    ll_spflow = log_likelihood(spn_copy, x_np)

    torch_spn_model.eval()
    with torch.no_grad():
        ll_torch = torch_spn_model(x_torch)

    ll_torch_np = ll_torch.detach().cpu().numpy()

    print("x:")
    print(x_np)
    print("SPFlow log-likelihood:")
    print(ll_spflow)
    print("Torch SPN log-likelihood:")
    print(ll_torch_np)

    print("Diff = Torch - SPFlow:")
    print(ll_torch_np - ll_spflow)


if __name__ == "__main__":
    main()