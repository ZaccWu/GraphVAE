from __future__ import division
from src.initializations import weightVariableGlorot
import tensorflow as tf

LAYERUIDS = {} # Global unique layer ID dictionary for layer name assignment

def getLayerUid(layerName =''):
    """Helper function, assigns unique layer IDs """
    if layerName not in LAYERUIDS:
        LAYERUIDS[layerName] = 1
        return 1
    else:
        LAYERUIDS[layerName] += 1
        return LAYERUIDS[layerName]

def dropoutSparse(x, keepProb, numNonzeroElems):
    """Dropout for sparse tensors """
    noiseShape = [numNonzeroElems]
    randomTensor = keepProb
    randomTensor += tf.random_uniform(noiseShape)
    dropoutMask = tf.cast(tf.floor(randomTensor), dtype=tf.bool)
    preOut = tf.sparse_retain(x, dropoutMask)
    return preOut * (1. / keepProb)

class GraphConvolution():
    """ Graph convolution layer """
    def __init__(self, inputDim, outputDim, adj, dropout = 0., act = tf.nn.relu):
        self.layerName = self.__class__.__name__.lower()
        self.name = self.layerName + '_' + str(getLayerUid(self.layerName))
        self.vars = {}

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weightVariableGlorot(inputDim, outputDim, name ="weights")
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
    def __init__(self, inputDim, outputDim, adj, featuresNonzero, dropout = 0., act = tf.nn.relu):
        self.layerName = self.__class__.__name__.lower()
        self.name = self.layerName + '_' + str(getLayerUid(self.layerName))
        self.vars = {}

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weightVariableGlorot(inputDim, outputDim, name ="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.featuresNonzero = featuresNonzero

    def __call__(self, inputs):
        x = inputs
        x = dropoutSparse(x, 1 - self.dropout, self.featuresNonzero)
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




