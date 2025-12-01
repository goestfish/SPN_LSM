import torch
import torch.nn as nn
import numpy as np

from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical
from spn.structure.Base import Sum, Product, Leaf, get_topological_order
from spn.algorithms.TransformStructure import Copy

import torch_spn_layers as SPN_lay


_node_log_torch_graph = {
    Sum:      SPN_lay.log_sum_to_torch_graph,
    Product:  SPN_lay.log_prod_to_torch_graph,
    Gaussian: SPN_lay.gaussian_to_torch_graph,
    Categorical: SPN_lay.categorical_to_torch_graph,
}



def torch_graph_to_sum(node, torchvar):

    with torch.no_grad():
        weights = torch.softmax(torchvar, dim=0).cpu().numpy().tolist()
    node.weights = weights


def torch_graph_to_gaussian(node, torchvars):
    mean_param, stdev_param = torchvars
    with torch.no_grad():
        node.mean = mean_param.cpu().numpy()
        node.stdev = np.maximum(stdev_param.cpu().numpy(), 0.01)


def torch_graph_to_categorical(node, torchvar):
    with torch.no_grad():
        p = torch.softmax(torchvar, dim=0).cpu().numpy()
        # 加 1e-3 防止 0
        node.p = (p + 1e-3).tolist()


_torch_graph_to_node = {
    Sum:        torch_graph_to_sum,
    Gaussian:   torch_graph_to_gaussian,
    Categorical: torch_graph_to_categorical,
}


def spn_to_torch_graph(node,
                       torch_input,
                       eval_functions=_node_log_torch_graph,
                       **args):

    all_results = {}
    nodes = get_topological_order(node)
    for node_type, func in eval_functions.items():
        if "_eval_func" not in node_type.__dict__:
            node_type._eval_func = []
        node_type._eval_func.append(func)
        node_type._is_leaf = issubclass(node_type, Leaf)

    leaf_func = eval_functions.get(Leaf, None)

    tmp_children_list = []
    len_tmp_children_list = 0

    for n_idx, n in enumerate(nodes):
        try:
            func = n.__class__._eval_func[-1]
            n_is_leaf = n.__class__._is_leaf
        except Exception:
            if isinstance(n, Leaf) and leaf_func is not None:
                func = leaf_func
                n_is_leaf = True
            else:
                raise AssertionError(
                    "No eval function associated with type: %s" % (n.__class__.__name__)
                )

        if n_is_leaf:
            result = func(n, torch_input, **args)
        else:
            len_children = len(n.children)
            if len_tmp_children_list < len_children:
                tmp_children_list.extend([None] * len_children)
                len_tmp_children_list = len(tmp_children_list)
            for i in range(len_children):
                ci = n.children[i]
                tmp_children_list[i] = all_results[ci]

            result = func(n, tmp_children_list[0:len_children], **args)

        all_results[n] = result

    for node_type, func in eval_functions.items():
        del node_type._eval_func[-1]
        if len(node_type._eval_func) == 0:
            delattr(node_type, "_eval_func")

    torch_graph = all_results[node]
    return torch_graph


class TorchSPN(nn.Module):

    def __init__(self, spn_root, eval_functions=_node_log_torch_graph, trainable_leaf=True):
        super().__init__()
        self.spn_copy = Copy(spn_root)
        self.eval_functions = eval_functions
        self.trainable_leaf = trainable_leaf

        self.variable_dict = {}

    def forward(self, x):
        log_prob = spn_to_torch_graph(
            self.spn_copy,
            x,
            eval_functions=self.eval_functions,
            variable_dict=self.variable_dict,
            trainable_leaf=self.trainable_leaf,
        )
        return log_prob


def create_torch_spn(spn):

    model = TorchSPN(spn)
    return model, model.variable_dict, model.spn_copy

def create_torch_spn_parts(spn_x, label_ids, trainable_leaf):

    all_spn_x_y = []
    all_spn_x_y_dicts = []
    all_prior = []
    all_spn_x_y_models = []

    spn_x_copy = Copy(spn_x)

    sorted_list = list(sorted(
        zip(spn_x_copy.children, spn_x_copy.weights, label_ids),
        key=lambda x: x[2]
    ))
    print('SPN ROOT weights:', spn_x_copy.weights)

    for spn_x_y, prior_y, label_id in sorted_list:
        model = TorchSPN(
            spn_x_y,
            eval_functions=_node_log_torch_graph,
            trainable_leaf=trainable_leaf,
        )
        all_spn_x_y.append(spn_x_y)
        all_spn_x_y_dicts.append(model.variable_dict)
        all_prior.append(prior_y)
        all_spn_x_y_models.append(model)

    return spn_x_copy, all_spn_x_y, all_spn_x_y_dicts, all_prior, all_spn_x_y_models

def torch_graph_to_spn(variable_dict, torch_graph_to_node=_torch_graph_to_node):

    tensors = []

    for n, torchvars in variable_dict.items():
        tensors.append(torchvars)

    for i, (n, torchvars) in enumerate(variable_dict.items()):
        fn = torch_graph_to_node.get(type(n), None)
        if fn is not None:
            fn(n, tensors[i])

def build_torch_spn(root, num_vars=None, device="cpu", trainable_leaf=True):
    model = TorchSPN(
        spn_root=root,
        eval_functions=_node_log_torch_graph,
        trainable_leaf=trainable_leaf,
    )
    model.to(device)
    return model