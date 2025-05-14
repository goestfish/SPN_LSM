import tensorflow.compat.v2 as tf
print(tf.__version__)
import numpy as np
from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.Base import Sum, Product, Leaf,get_topological_order
from spn.algorithms.TransformStructure import Copy
import tf_SPN_layers as SPN_lay


_node_log_tf_graph = {Sum: SPN_lay.log_sum_to_tf_graph, Product:  SPN_lay.log_prod_to_tf_graph,Gaussian:SPN_lay.gaussian_to_tf_graph,Categorical:SPN_lay.categorical_to_tf_graph}

def tf_graph_to_sum(node, tfvar):
    weights=tf.nn.softmax(tfvar).numpy().tolist()
    node.weights = weights

def tf_graph_to_gaussian(node, tfvar):
    node.mean = tfvar[0].numpy()
    node.stdev = np.maximum(tfvar[1].numpy(),0.01)

def tf_graph_to_categorical(node, tfvar):
    node.p= (tf.nn.softmax(tfvar).numpy()+1e-3).tolist()


_tf_graph_to_node = {Sum: tf_graph_to_sum,Gaussian:tf_graph_to_gaussian,Categorical:tf_graph_to_categorical}

def spn_to_tf_graph(node,tf_input, eval_functions=_node_log_tf_graph, **args):


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

    for n_idx,n in enumerate(nodes):

        try:
            func = n.__class__._eval_func[-1]
            n_is_leaf = n.__class__._is_leaf
        except:
            if isinstance(n, Leaf) and leaf_func is not None:
                func = leaf_func
                n_is_leaf = True
            else:
                raise AssertionError("No lambda function associated with type: %s" % (n.__class__.__name__))

        if n_is_leaf:
            result = func(n,tf_input, **args)#(tf_input)
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
    #'''
    for node_type, func in eval_functions.items():
        del node_type._eval_func[-1]
        if len(node_type._eval_func) == 0:
            delattr(node_type, "_eval_func")
    #'''
    tf_graph=all_results[node]

    return tf_graph

def create_tf_spn(spn, data_shape):
    # Make sure, that the passed SPN is not modified
    spn_copy = Copy(spn)
    variable_dict = {}
    tf_input = tf.keras.Input(shape=data_shape)
    graph = spn_to_tf_graph(spn_copy, tf_input, variable_dict=variable_dict)
    return tf.keras.Model(inputs=tf_input, outputs=graph),variable_dict,spn_copy

def create_tf_spn_parts(spn_x, data_shape,label_ids,trainable_leaf):
    # Make sure, that the passed SPN is not modified
    all_spn_x_y,all_spn_x_y_dicts,all_prior,all_spn_x_y_model=[],[],[],[]
    spn_x_copy = Copy(spn_x)

    sorted_list=list(sorted(zip(spn_x.children,spn_x.weights,label_ids),key=lambda x:x[2]))
    print('SPN ROOT weights:',spn_x.weights)
    for spn_x_y,prior_y,label_id in sorted_list:
        variable_dict = {}
        tf_input = tf.keras.Input(shape=data_shape)
        graph = spn_to_tf_graph(spn_x_y, tf_input, variable_dict=variable_dict,trainable_leaf=trainable_leaf)
        spn_model=tf.keras.Model(inputs=tf_input, outputs=graph,name='label_id'+str(int(label_id)))
        all_spn_x_y.append(spn_x_y)
        all_spn_x_y_dicts.append(variable_dict)
        all_prior.append(prior_y)
        all_spn_x_y_model.append(spn_model)


    return spn_x_copy,all_spn_x_y,all_spn_x_y_dicts,all_prior,all_spn_x_y_model#,tf_spn_root
def tf_graph_to_spn(variable_dict, tf_graph_to_node=_tf_graph_to_node):
    tensors = []

    for n, tfvars in variable_dict.items():
        tensors.append(tfvars)

    for i, (n, tfvars) in enumerate(variable_dict.items()):
        tf_graph_to_node[type(n)](n, tensors[i])#variable_list[i]

