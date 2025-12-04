import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

class GaussianLayer(nn.Module):

    def __init__(self, mean, stdev,
                 dtype=torch.float32,
                 log_space=True,
                 trainable_nodes=True,
                 name="Gauss"):
        super().__init__()

        mean_t = torch.as_tensor(mean, dtype=dtype)
        stdev_t = torch.as_tensor(stdev, dtype=dtype)
        req_grad = bool(trainable_nodes)
        self.mean = nn.Parameter(mean_t, requires_grad=req_grad)
        self.stdev = nn.Parameter(stdev_t, requires_grad=req_grad)
        tmp_t = torch.tensor(0.001, dtype=dtype)
        self.tmp = nn.Parameter(tmp_t, requires_grad=req_grad)
        self.trainable_nodes = req_grad

    def forward(self, inputs, nd_idxs):

        device = inputs.device
        x = inputs[nd_idxs[:, 0], nd_idxs[:, 1]]
        x = x.view(inputs.size(0), 1)

        mean = self.mean.to(device)
        tmp = self.tmp.to(device)
        stdev_param = self.stdev.to(device)

        if self.trainable_nodes:
            stdev = torch.maximum(stdev_param, tmp)
        else:
            stdev = stdev_param

        dist = Normal(loc=mean, scale=stdev)
        log_prob = dist.log_prob(x)

        return log_prob  # [B, 1]

class CategoricalLayer(nn.Module):

    def __init__(self, prob, log_space=True, name="Categorical"):
        super().__init__()
        prob_t = torch.as_tensor(prob, dtype=torch.float32)
        self.probs = nn.Parameter(prob_t, requires_grad=False)

    def forward(self, inputs, nd_idxs):
        device = inputs.device
        x = inputs[nd_idxs[:, 0], nd_idxs[:, 1]]  # [B]
        x = x.view(inputs.size(0), 1)

        logits = self.probs.to(device)

        softmax_probs = F.softmax(logits, dim=0)
        dist = Categorical(probs=softmax_probs)
        x_idx = x.squeeze(-1).long()

        log_prob = dist.log_prob(x_idx)  # [B]

        return log_prob.view(-1, 1)  # [B, 1]


def get_batch_idx(node, data_input):

    device = data_input.device
    batch_size = data_input.size(0)

    idx = node.scope[0]
    batch_idxs = torch.arange(batch_size, device=device, dtype=torch.long)
    feat_idxs = torch.full((batch_size,), idx,
                           device=device, dtype=torch.long)

    nd_idxs = torch.stack([batch_idxs, feat_idxs], dim=1)  # [B, 2]
    return nd_idxs


class LogSumLayer(nn.Module):
    
    def __init__(self, softmax_inverse, dtype=torch.float32, name="log_sum"):
        super().__init__()
        logits = torch.as_tensor(softmax_inverse, dtype=dtype)
        self.logits = nn.Parameter(logits, requires_grad=True)

    def forward(self, inputs):
        device = inputs[0].device

        weights = F.softmax(self.logits.to(device), dim=0)
        log_w = torch.log(weights + 1e-8).to(device)

        children_prob = torch.stack(inputs, dim=1)
        log_w = log_w.view(1, -1, 1)

        out = torch.logsumexp(children_prob + log_w, dim=1)
        return out


class LogProdLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs):

        # tf.add_n(inputs)
        out = inputs[0]
        for x in inputs[1:]:
            out = out + x
        return out


def gaussian_to_torch_graph(node,
                            data_input=None,
                            log_space=True,
                            variable_dict=None,
                            dtype=torch.float32,
                            trainable_leaf=True):

    nd_idxs = get_batch_idx(node, data_input)
    layer = GaussianLayer(node.mean, node.stdev,
                          dtype=dtype,
                          trainable_nodes=trainable_leaf,
                          name=node.__class__.__name__ + str(node.id))
    if variable_dict is not None:
        variable_dict[node] = (layer.mean, layer.stdev)
    return layer(data_input, nd_idxs)


def categorical_to_torch_graph(node,
                               data_placeholder=None,
                               log_space=True,
                               variable_dict=None,
                               dtype=np.float32,
                               trainable_leaf=False):
    
    nd_idxs = get_batch_idx(node, data_placeholder)
    p = np.array(node.p, dtype=dtype)

    if p.sum() > 0:
        p = p / p.sum()
    eps = 1e-8
    p = np.clip(p, eps, 1.0)
    softmax_inverse = np.log(p / np.max(p)).astype(dtype)
    layer = CategoricalLayer(softmax_inverse,
                             log_space,
                             name=node.__class__.__name__ + str(node.id))
    
    if variable_dict is not None:
        variable_dict[node] = layer.probs

    return layer(data_placeholder, nd_idxs)


def log_prod_to_torch_graph(node,
                            children,
                            data_placeholder=None,
                            variable_dict=None,
                            log_space=True,
                            dtype=np.float32,
                            trainable_leaf=False):
    assert log_space
    layer = LogProdLayer()
    return layer(children)


def log_sum_to_torch_graph(node,
                           children,
                           data_placeholder=None,
                           variable_dict=None,
                           log_space=True,
                           dtype=np.float32,
                           trainable_leaf=False):
    assert log_space
    softmax_inverse = np.log(node.weights / np.max(node.weights)).astype(dtype)
    layer = LogSumLayer(softmax_inverse,
                        name=node.__class__.__name__ + str(node.id))
    if variable_dict is not None:
        variable_dict[node] = layer.logits
    return layer(children)

if __name__ == "__main__":
    class DummyNode:
        def __init__(self, idx, mean, stdev, weights=None, p=None, node_id=0):
            self.scope = [idx]
            self.mean = mean
            self.stdev = stdev
            self.weights = weights
            self.p = p
            self.id = node_id

    B, D = 4, 3
    x = torch.randn(B, D)

    node_g = DummyNode(idx=1, mean=0.0, stdev=1.0, node_id=1)
    var_dict = {}
    logp_g = gaussian_to_torch_graph(node_g, x, variable_dict=var_dict)
    print("Gaussian logp:", logp_g.shape, logp_g)

    node_s = DummyNode(idx=0, mean=0.0, stdev=1.0, weights=np.array([0.3, 0.7]), node_id=2)
    child1 = torch.randn(B, 1)
    child2 = torch.randn(B, 1)
    logp_sum = log_sum_to_torch_graph(node_s, [child1, child2])
    print("Log-sum:", logp_sum.shape, logp_sum)