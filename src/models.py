from src.layers import *
import tensorflow as tf

class GCNModelAE():
    """
    Standard Graph Autoencoder from Kipf and Welling (2016),
    with 2-layer GCN encoder and inner product decoder
    """
    def __init__(self, param, placeholders, num_features, features_nonzero):
        self.params = param
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']

        self.hidden = GraphConvolutionSparse(input_dim = self.input_dim,
                                             output_dim = self.params['hidden'],
                                             adj = self.adj,
                                             features_nonzero = self.features_nonzero,
                                             act = tf.nn.relu,
                                             dropout = self.dropout)(self.inputs)

        self.z_mean = GraphConvolution(input_dim = self.params['hidden'],
                                       output_dim = self.params['dimension'],
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout)(self.hidden)

        self.reconstructions = InnerProductDecoder(act = lambda x: x)(self.z_mean)

    def __call__(self):
        pass


class GCNModelVAE():
    """
    Standard Graph Variational Autoencoder from Kipf and Welling (2016),
    with 2-layer GCN encoder, Gaussian distributions and inner product decoder
    """
    def __init__(self, param, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        self.params = param
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']

        self.hidden = GraphConvolutionSparse(input_dim = self.input_dim,
                                             output_dim = self.params['hidden'],
                                             adj = self.adj,
                                             features_nonzero = self.features_nonzero,
                                             act = tf.nn.relu,
                                             dropout = self.dropout)(self.inputs)

        self.z_mean = GraphConvolution(input_dim = self.params['hidden'],
                                       output_dim = self.params['dimension'],
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout)(self.hidden)

        self.z_log_std = GraphConvolution(input_dim = self.params['hidden'],
                                          output_dim = self.params['dimension'],
                                          adj = self.adj,
                                          act = lambda x: x,
                                          dropout = self.dropout)(self.hidden)

        self.z = self.z_mean + tf.random_normal([self.n_samples, self.params['dimension']]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(act = lambda x: x)(self.z)

    def __call__(self):
        pass

class LinearModelAE():
    """
    Linear Graph Autoencoder, as defined in Section 3 of NeurIPS 2019 workshop paper,
    with linear encoder and inner product decoder
    """
    def __init__(self, param, placeholders, num_features, features_nonzero, **kwargs):
        self.params = param
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']

        self.z_mean = GraphConvolutionSparse(input_dim = self.input_dim,
                                             output_dim = self.params['dimension'],
                                             adj = self.adj,
                                             features_nonzero = self.features_nonzero,
                                             act = lambda x: x,
                                             dropout = self.dropout)(self.inputs)

        self.reconstructions = InnerProductDecoder(act = lambda x: x)(self.z_mean)

    def __call__(self):
        pass

class LinearModelVAE():
    """
    Linear Graph Variational Autoencoder, as defined in Section 3 of
    NeurIPS 2019 workshop paper, with Gaussian distributions, linear
    encoders for mu and sigma, and inner product decoder
    """
    def __init__(self, param, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        self.params = param
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']

        self.z_mean = GraphConvolutionSparse(input_dim = self.input_dim,
                                             output_dim = self.params['dimension'],
                                             adj = self.adj,
                                             features_nonzero=self.features_nonzero,
                                             act = lambda x: x,
                                             dropout = self.dropout)(self.inputs)

        self.z_log_std = GraphConvolutionSparse(input_dim = self.input_dim,
                                                output_dim = self.params['dimension'],
                                                adj = self.adj,
                                                features_nonzero = self.features_nonzero,
                                                act = lambda x: x,
                                                dropout = self.dropout)(self.inputs)

        self.z = self.z_mean + tf.random_normal([self.n_samples, self.params['dimension']]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(act = lambda x: x)(self.z)

    def __call__(self):
        pass


class DeepGCNModelAE():
    """
    "Deep" Graph Autoencoder from Section 4 of NeurIPS 2019 workshop paper,
    with 3-layer GCN encoder, and inner product decoder
    """
    def __init__(self, param, placeholders, num_features, features_nonzero, **kwargs):
        self.params = param
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']

        self.hidden1 = GraphConvolutionSparse(input_dim = self.input_dim,
                                              output_dim = self.params['hidden'],
                                              adj = self.adj,
                                              features_nonzero = self.features_nonzero,
                                              act = tf.nn.relu,
                                              dropout = self.dropout)(self.inputs)

        self.hidden2 = GraphConvolution(input_dim = self.params['hidden'],
                                        output_dim = self.params['hidden'],
                                        adj = self.adj,
                                        act = tf.nn.relu,
                                        dropout = self.dropout)(self.hidden1)

        self.z_mean = GraphConvolution(input_dim = self.params['hidden'],
                                       output_dim = self.params['dimension'],
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout)(self.hidden2)

        self.reconstructions = InnerProductDecoder(act = lambda x: x)(self.z_mean)

    def __call__(self):
        pass


class DeepGCNModelVAE():
    """
    "Deep" Graph Variational Autoencoder, from Section 4 of NeurIPS 2019
    workshop paper, with Gaussian distributions, 3-layer GCN encoders for
    mu and sigma, and inner product decoder
    """
    def __init__(self, param, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        self.params = param
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']

        self.hidden1 = GraphConvolutionSparse(input_dim = self.input_dim,
                                              output_dim = self.params['hidden'],
                                              adj = self.adj,
                                              features_nonzero = self.features_nonzero,
                                              act = tf.nn.relu,
                                              dropout = self.dropout)(self.inputs)

        self.hidden2 = GraphConvolution(input_dim = self.params['hidden'],
                                        output_dim = self.params['hidden'],
                                        adj = self.adj,
                                        act = tf.nn.relu,
                                        dropout = self.dropout)(self.hidden1)

        self.z_mean = GraphConvolution(input_dim = self.params['hidden'],
                                       output_dim = self.params['dimension'],
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout)(self.hidden2)

        self.z_log_std = GraphConvolution(input_dim = self.params['hidden'],
                                          output_dim = self.params['dimension'],
                                          adj = self.adj,
                                          act = lambda x: x,
                                          dropout = self.dropout)(self.hidden2)

        self.z = self.z_mean + tf.random_normal([self.n_samples, self.params['dimension']]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(act = lambda x: x)(self.z)

    def __call__(self):
        pass