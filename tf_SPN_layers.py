import tensorflow.compat.v2 as tf
import numpy as np
from tensorflow_probability import distributions as tfd

class Gaussian_layer(tf.keras.layers.Layer):
    def __init__(self,mean,stdev,dtype=tf.dtypes.float32,log_space=True,trainable_nodes=True,name='Gauss'):#(self,node, data_input=None, log_space=True, variable_dict=None, dtype=tf.dtypes.float32):
        super(Gaussian_layer,self).__init__()
        # with tf.name_scope("%s_%s" % (node.__class__.__name__, node.id)):
        self.mean = tf.Variable(initial_value=mean, dtype=dtype, trainable=trainable_nodes,name=name+'mean')  # name="mean",
        self.stdev = tf.Variable(initial_value=stdev, dtype=dtype, trainable=trainable_nodes,name=name+'std')  # name="stdev",
        self.tmp=tf.Variable(initial_value= 0.001, dtype=dtype, trainable=trainable_nodes,name=name+'tmp')
        self.trainable_nodes=trainable_nodes
    def call(self, inputs,nd_idxs):
        new_input = tf.gather_nd(inputs, nd_idxs)
        new_input = tf.reshape(new_input, (tf.shape(inputs)[0], 1))
        if self.trainable_nodes:
            stdev = tf.keras.layers.maximum([self.stdev, self.tmp])
        else:
            stdev=self.stdev
        self.dist = tfd.Normal(loc=self.mean, scale=stdev)
        result = self.dist.log_prob(new_input)

        return result
class Categorical_layer(tf.keras.layers.Layer):#softmaxInverse
    def __init__(self, prob, log_space=True,name='Categorical'):
        super(Categorical_layer,self).__init__()

        self.probs = tf.Variable(initial_value=prob, trainable=False, dtype=tf.dtypes.float32, name=name)


    def call(self, inputs,nd_idxs):
        new_input = tf.gather_nd(inputs, nd_idxs)
        new_input = tf.reshape(new_input, (tf.shape(inputs)[0], 1))

        softmax_probs=tf.nn.softmax(self.probs)
        return tfd.Categorical(probs=softmax_probs,dtype=tf.dtypes.float32,allow_nan_stats=False).log_prob(new_input)
def gaussian_to_tf_graph(node, data_input=None, log_space=True, variable_dict=None, dtype=tf.dtypes.float32,trainable_leaf=True):
    nd_idxs=get_batch_idx(node,data_input)
    layer=Gaussian_layer(node.mean,node.stdev,trainable_nodes=trainable_leaf,name=node.__class__.__name__+str(node.id))
    variable_dict[node] = (layer.mean, layer.stdev)
    return layer(data_input,nd_idxs)


def categorical_to_tf_graph(node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32,trainable_leaf=False):
    nd_idxs = get_batch_idx(node, data_placeholder)
    p = np.array(node.p, dtype=dtype)

    softmaxInverse = np.log(p / np.max(p)).astype(dtype)

    layer = Categorical_layer(softmaxInverse, log_space, name=node.__class__.__name__ + str(node.id))
    variable_dict[node] = layer.probs
    return layer(data_placeholder,nd_idxs)


def get_batch_idx(node,data_input):
    idxs = [node.scope[0]]
    idxs = tf.constant(idxs, tf.int32)
    batch_idxs = tf.range(tf.shape(data_input)[0])#tf.math.floordiv(tf.range(0, batch_size * num_idxs), num_idxs)[:, None]
    idx_list=tf.tile(idxs,tf.shape(data_input)[0:1])
    batch_idxs=tf.expand_dims(batch_idxs,axis=1)
    idx_list = tf.expand_dims(idx_list, axis=1)
    nd_idxs = tf.concat([batch_idxs,idx_list ], axis=1)
    #print('DEBUG', node.__class__.__name__ + str(node.id), 'scope=', node.scope)
    return nd_idxs

class Log_sum_layer(tf.keras.layers.Layer):
    def __init__(self, softmax_inverse,dtype=tf.dtypes.float32,name='log_sum'):#, children, data_placeholder=None, variable_dict=None, log_space=True, dtype=np.float32):
        super(Log_sum_layer,self).__init__()

        self.tf_weights=tf.Variable(initial_value=softmax_inverse, dtype=dtype, trainable=True,name=name)



    def call(self, inputs):
        tf_weights = tf.nn.softmax(self.tf_weights)
        tf_weights=tf.expand_dims(tf.expand_dims(tf_weights,axis=0),axis=-1)
        # Stack children along axis 1
        children_prob = tf.stack(inputs, axis=1)

        out=tf.reduce_logsumexp(children_prob + tf.math.log(tf_weights), axis=1)

        return out
class Log_prod_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(Log_prod_layer,self).__init__()

    def call(self, inputs):
        out=tf.add_n(inputs)

        return out





def log_prod_to_tf_graph(node, children, data_placeholder=None, variable_dict=None, log_space=True, dtype=np.float32,trainable_leaf=False):
    assert log_space
    layer=Log_prod_layer()
    return layer(children)
def log_sum_to_tf_graph(node, children, data_placeholder=None, variable_dict=None, log_space=True, dtype=np.float32,trainable_leaf=False):
    assert log_space
    softmax_inverse = np.log(node.weights / np.max(node.weights)).astype(dtype)
    layer=Log_sum_layer(softmax_inverse,name=node.__class__.__name__+str(node.id))
    variable_dict[node] = layer.tf_weights


    return  layer(children)



