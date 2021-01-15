from src.networkUnit import *
import tensorflow as tf

class gcnAE():
    """
    Standard Graph Autoencoder from Kipf and Welling (2016),
    with 2-layer GCN encoder and inner product decoder
    """
    def __init__(self, param, placeholders, numFeatures, featuresNonzero):
        self.params = param
        self.inputs = placeholders['features']
        self.inputDim = numFeatures
        self.featuresNonzero = featuresNonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']

        self.hidden = GraphConvolutionSparse(inputDim= self.inputDim,
                                             outputDim= self.params['hidden'],
                                             adj = self.adj,
                                             featuresNonzero= self.featuresNonzero,
                                             act = tf.nn.relu,
                                             dropout = self.dropout)(self.inputs)

        self.zMean = GraphConvolution(inputDim= self.params['hidden'],
                                      outputDim= self.params['dimension'],
                                      adj = self.adj,
                                      act = lambda x: x,
                                      dropout = self.dropout)(self.hidden)

        self.reconstructions = InnerProductDecoder(act = lambda x: x)(self.zMean)

    def __call__(self):
        pass


class gcnVAE():
    """
    Standard Graph Variational Autoencoder from Kipf and Welling (2016),
    with 2-layer GCN encoder, Gaussian distributions and inner product decoder
    """
    def __init__(self, param, placeholders, numFeatures, numNodes, featuresNonzero):
        self.params = param
        self.inputs = placeholders['features']
        '''
        self.inputs: 
        SparseTensor(indices=Tensor("Placeholder_2:0", shape=(?, ?), dtype=int64), 
            values=Tensor("Placeholder_1:0", shape=(?,), dtype=float32), 
            dense_shape=Tensor("Placeholder:0", shape=(?,), dtype=int64))
        '''
        self.inputDim = numFeatures
        self.featuresNonzero = featuresNonzero
        self.nSamples = numNodes
        # cora:
            # inputdim: 1433
            # featuresNonzero: 49216
            # nSamples: 2708
        # citeseer:
            # inputdim: 3703
            # featuresNonzero: 105165 (feature=False: 3327)
            # nSamples: 3327
        self.adj = placeholders['adj']
        '''
        self.adj:
        SparseTensor(indices=Tensor("Placeholder_5:0", shape=(?, ?), dtype=int64),
             values=Tensor("Placeholder_4:0", shape=(?,), dtype=float32),
             dense_shape=Tensor("Placeholder_3:0", shape=(?,), dtype=int64))
        '''
        self.dropout = placeholders['dropout']
        # self.dropout: Tensor("PlaceholderWithDefault:0", shape=(), dtype=float32)
        self.hidden = GraphConvolutionSparse(inputDim= self.inputDim,
                                             outputDim= self.params['hidden'],
                                             adj = self.adj,
                                             featuresNonzero= self.featuresNonzero,
                                             act = tf.nn.relu,
                                             dropout = self.dropout)(self.inputs)

        self.zMean = GraphConvolution(inputDim= self.params['hidden'],
                                      outputDim= self.params['dimension'],
                                      adj = self.adj,
                                      act = lambda x: x,
                                      dropout = self.dropout)(self.hidden)

        self.zLogStd = GraphConvolution(inputDim= self.params['hidden'],
                                        outputDim= self.params['dimension'],
                                        adj = self.adj,
                                        act = lambda x: x,
                                        dropout = self.dropout)(self.hidden)

        self.z = self.zMean + tf.random_normal([self.nSamples, self.params['dimension']]) * tf.exp(self.zLogStd)
        self.reconstructions = InnerProductDecoder(act = lambda x: x)(self.z)

    def __call__(self):
        pass

class linearAE():
    """
    Linear Graph Autoencoder, as defined in Section 3 of NeurIPS 2019 workshop paper,
    with linear encoder and inner product decoder
    """
    def __init__(self, param, placeholders, numFeatures, featuresNonzero):
        self.params = param
        self.inputs = placeholders['features']
        self.inputDim = numFeatures
        self.featuresNonzero = featuresNonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']

        self.zMean = GraphConvolutionSparse(inputDim= self.inputDim,
                                            outputDim= self.params['dimension'],
                                            adj = self.adj,
                                            featuresNonzero= self.featuresNonzero,
                                            act = lambda x: x,
                                            dropout = self.dropout)(self.inputs)

        self.reconstructions = InnerProductDecoder(act = lambda x: x)(self.zMean)

    def __call__(self):
        pass

class linearVAE():
    """
    Linear Graph Variational Autoencoder, as defined in Section 3 of
    NeurIPS 2019 workshop paper, with Gaussian distributions, linear
    encoders for mu and sigma, and inner product decoder
    """
    def __init__(self, param, placeholders, numFeatures, numNodes, featuresNonzero):
        self.params = param
        self.inputs = placeholders['features']
        self.inputDim = numFeatures
        self.featuresNonzero = featuresNonzero
        self.nSamples = numNodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']

        self.zMean = GraphConvolutionSparse(inputDim= self.inputDim,
                                            outputDim= self.params['dimension'],
                                            adj = self.adj,
                                            featuresNonzero=self.featuresNonzero,
                                            act = lambda x: x,
                                            dropout = self.dropout)(self.inputs)

        self.zLogStd = GraphConvolutionSparse(inputDim= self.inputDim,
                                              outputDim= self.params['dimension'],
                                              adj = self.adj,
                                              featuresNonzero= self.featuresNonzero,
                                              act = lambda x: x,
                                              dropout = self.dropout)(self.inputs)

        self.z = self.zMean + tf.random_normal([self.nSamples, self.params['dimension']]) * tf.exp(self.zLogStd)

        self.reconstructions = InnerProductDecoder(act = lambda x: x)(self.z)

    def __call__(self):
        pass


class gcnDeepAE():
    """
    "Deep" Graph Autoencoder from Section 4 of NeurIPS 2019 workshop paper,
    with 3-layer GCN encoder, and inner product decoder
    """
    def __init__(self, param, placeholders, numFeatures, featuresNonzero):
        self.params = param
        self.inputs = placeholders['features']
        self.inputDim = numFeatures
        self.featuresNonzero = featuresNonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']

        self.hidden1 = GraphConvolutionSparse(inputDim= self.inputDim,
                                              outputDim= self.params['hidden'],
                                              adj = self.adj,
                                              featuresNonzero= self.featuresNonzero,
                                              act = tf.nn.relu,
                                              dropout = self.dropout)(self.inputs)

        self.hidden2 = GraphConvolution(inputDim= self.params['hidden'],
                                        outputDim= self.params['hidden'],
                                        adj = self.adj,
                                        act = tf.nn.relu,
                                        dropout = self.dropout)(self.hidden1)

        self.zMean = GraphConvolution(inputDim= self.params['hidden'],
                                      outputDim= self.params['dimension'],
                                      adj = self.adj,
                                      act = lambda x: x,
                                      dropout = self.dropout)(self.hidden2)

        self.reconstructions = InnerProductDecoder(act = lambda x: x)(self.zMean)

    def __call__(self):
        pass


class gcnDeepVAE():
    """
    "Deep" Graph Variational Autoencoder, from Section 4 of NeurIPS 2019
    workshop paper, with Gaussian distributions, 3-layer GCN encoders for
    mu and sigma, and inner product decoder
    """
    def __init__(self, param, placeholders, numFeatures, numNodes, featuresNonzero):
        self.params = param
        self.inputs = placeholders['features']
        self.inputDim = numFeatures
        self.featuresNonzero = featuresNonzero
        self.nSamples = numNodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']

        self.hidden1 = GraphConvolutionSparse(inputDim= self.inputDim,
                                              outputDim= self.params['hidden'],
                                              adj = self.adj,
                                              featuresNonzero= self.featuresNonzero,
                                              act = tf.nn.relu,
                                              dropout = self.dropout)(self.inputs)

        self.hidden2 = GraphConvolution(inputDim= self.params['hidden'],
                                        outputDim= self.params['hidden'],
                                        adj = self.adj,
                                        act = tf.nn.relu,
                                        dropout = self.dropout)(self.hidden1)

        self.zMean = GraphConvolution(inputDim= self.params['hidden'],
                                      outputDim= self.params['dimension'],
                                      adj = self.adj,
                                      act = lambda x: x,
                                      dropout = self.dropout)(self.hidden2)

        self.zLogStd = GraphConvolution(inputDim= self.params['hidden'],
                                        outputDim= self.params['dimension'],
                                        adj = self.adj,
                                        act = lambda x: x,
                                        dropout = self.dropout)(self.hidden2)

        self.z = self.zMean + tf.random_normal([self.nSamples, self.params['dimension']]) * tf.exp(self.zLogStd)

        self.reconstructions = InnerProductDecoder(act = lambda x: x)(self.z)

    def __call__(self):
        pass