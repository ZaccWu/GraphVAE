from src.networkUnit import *
import tensorflow as tf
from util.metrics import *

class gcnStdHVAE():
    """
    Standard Graph Variational Autoencoder from Kipf and Welling (2016),
    with 2-layer GCN encoder, Gaussian distributions and inner product decoder
    """
    def __init__(self, param, placeholders, numFeatures, numNodes, featuresNonzero):
        self.params = param
        self.inputs = placeholders['features']
        self.inputDim = numFeatures
        self.featuresNonzero = featuresNonzero
        self.nSamples = numNodes

        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']

        self.hidden = GraphConvolutionSparse(inputDim= self.inputDim,
                                             outputDim= self.params['hidden'],
                                             adj = self.adj,
                                             featuresNonzero= self.featuresNonzero,
                                             act = tf.nn.relu,
                                             dropout = self.dropout)(self.inputs)

        self.zMean = GraphConvolutionSparse(inputDim= self.inputDim,
                                             outputDim= self.params['dimension'],
                                             adj = self.adj,
                                             featuresNonzero= self.featuresNonzero,
                                             act = lambda x: x,
                                             dropout = self.dropout)(self.inputs)
        self.zLogStd = GraphConvolution(inputDim= self.params['hidden'],
                                        outputDim= self.params['dimension'],
                                        adj = self.adj,
                                        act = lambda x: x,
                                        dropout = self.dropout)(self.hidden)

        self.z = self.zMean + tf.random_normal([self.nSamples, self.params['dimension']]) * tf.exp(self.zLogStd)
        self.reconstructions = InnerProductDecoder(act = lambda x: x)(self.z)

        self.hsic = tf.reduce_sum(HSIC(self.params['dimension'])(self.z))

    def __call__(self):
        pass

