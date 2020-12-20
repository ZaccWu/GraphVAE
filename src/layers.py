from __future__ import division
from src.initializations import weightVariableGlorot
import tensorflow as tf

_LAYER_UIDS = {} # Global unique layer ID dictionary for layer name assignment

def getLayerUid(layer_name =''):
    """Helper function, assigns unique layer IDs """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def dropoutSparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

class GraphConvolution():
    """ Graph convolution layer """
    def __init__(self, input_dim, output_dim, adj, dropout = 0., act = tf.nn.relu):
        self.layerName = self.__class__.__name__.lower()
        self.name = self.layerName + '_' + str(getLayerUid(self.layerName))
        self.vars = {}

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weightVariableGlorot(input_dim, output_dim, name ="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1 - self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs"""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout = 0., act = tf.nn.relu):
        self.layerName = self.__class__.__name__.lower()
        self.name = self.layerName + '_' + str(getLayerUid(self.layerName))
        self.vars = {}

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weightVariableGlorot(input_dim, output_dim, name ="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        x = inputs
        x = dropoutSparse(x, 1 - self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class InnerProductDecoder():
    """Symmetric inner product decoder layer"""
    def __init__(self, dropout = 0., act = tf.nn.sigmoid):
        self.dropout = dropout
        self.act = act

    def __call__(self, inputs):
        inputs = tf.nn.dropout(inputs, 1 - self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs