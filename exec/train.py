from __future__ import division
from __future__ import print_function
from util.evaluation import get_roc_score, clustering_latent_space
from src.input_data import loadData, load_label
from src.kcore import computeKcore, expandEmbedding
from src.models import GCNModelAE, GCNModelVAE, LinearModelAE, LinearModelVAE, DeepGCNModelAE, DeepGCNModelVAE
from src.optimizer import OptimizerAE, OptimizerVAE
from src.preprocessing import *
import numpy as np
import os
import scipy.sparse as sp
import tensorflow as tf
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

param = {
    # select the dataset
    'dataset': 'cora',              # 'cora', 'citeseer', 'pubmed'
    # select the task
    'task': 'link_prediction',      # 'link_prediction', 'node_clustering'
    # select the model
    'model': 'linear_ae',              # 'gcn_ae', 'gcn_vae', 'linear_ae', 'linear_vae', 'deep_gcn_ae', 'deep_gcn_vae'
    # model parameters
    'dropout': 0.,                  # Dropout rate (1 - keep probability)
    'epochs': 200,
    'features': False,
    'learning_rate': 0.01,
    'hidden': 32,                   # Number of units in GCN hidden layer(s)
    'dimension': 16,                # Embedding dimension (Dimension of encoder output)
    # experimental parameters
    'nb_run': 1,                    # Number of model run + test
    'prop_val': 5.,                 # Proportion of edges in validation set (link prediction)
    'prop_test': 10.,               # Proportion of edges in test set (link prediction)
    'validation': False,            # Whether to report validation results at each epoch (link prediction)
    'verbose': True,                # Whether to print comments details
    # degeneracy framework parameters
    'kcore': False,                 # Whether to run k-core decomposition (False-train on the entire graph)
    'k': 2,
    'nb_iterations': 10,
}

# Lists to collect average results
if param['task'] == 'link_prediction':
    mean_roc = []
    mean_ap = []
elif param['task'] == 'node_clustering':
    mean_mutual_info = []
if param['kcore']:
    mean_time_kcore = []
    mean_time_train = []
    mean_time_expand = []
    mean_core_size = []
mean_time = []

# Load graph dataset
if param['verbose']:
    print("Loading data...")
adj_init, features_init = loadData(param['dataset'])
# Load ground-truth labels for node clustering task
if param['task'] == 'node_clustering':
    labels = load_label(param['dataset'])

###### Run the Experiments ######
# The entire training+test process is repeated nb_run times
for i in range(param['nb_run']):
    if param['task'] == 'link_prediction' :
        if param['verbose']:
            print("Masking test edges...")
        # Edge Masking for Link Prediction: compute Train/Validation/Test set
        adj, valEdges, valEdgesFalse, testEdges, testEdgesFalse = mask_test_edges(adj_init, param['prop_test'], param['prop_val'])
    elif param['task'] == 'node_clustering':
        adj_tri = sp.triu(adj_init)
        adj = adj_tri + adj_tri.T
    else:
        raise ValueError('Undefined task!')

    # Start computation of running times
    t_start = time.time()

    # Degeneracy Framework / K-Core Decomposition
    if param['kcore']:
        if param['verbose']:
            print("Starting k-core decomposition of the graph")
        # Save adjacency matrix of un-decomposed graph
        # (needed to embed nodes that are not in k-core, after GAE training)
        adj_orig = adj
        # Get the (smaller) adjacency matrix of the k-core subgraph,
        # and the corresponding nodes
        adj, nodes_kcore = computeKcore(adj, param['k'])
        # Get the (smaller) feature matrix of the nb_core graph
        if param['features']:
            features = features_init[nodes_kcore,:]
        # Flag to compute k-core decomposition's running time
        t_core = time.time()
    elif param['features']:
        features = features_init

    # Preprocessing and initialization
    if param['verbose']:
        print("Preprocessing and Initializing...")
    # Compute number of nodes
    num_nodes = adj.shape[0]
    # If features are not used, replace feature matrix by identity matrix
    if not param['features']:
        features = sp.identity(adj.shape[0])
    # Preprocessing on node features
    features = sparse_to_tuple(features)
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape = ())
    }

    # Create model
    model = None
    if param['model'] == 'gcn_ae':
        # Standard Graph Autoencoder
        model = GCNModelAE(param, placeholders, num_features, features_nonzero)
    elif param['model'] == 'gcn_vae':
        # Standard Graph Variational Autoencoder
        model = GCNModelVAE(param, placeholders, num_features, num_nodes,
                            features_nonzero)
    elif param['model'] == 'linear_ae':
        # Linear Graph Autoencoder
        model = LinearModelAE(param, placeholders, num_features, features_nonzero)
    elif param['model'] == 'linear_vae':
        # Linear Graph Variational Autoencoder
        model = LinearModelVAE(param, placeholders, num_features, num_nodes,
                               features_nonzero)
    elif param['model'] == 'deep_gcn_ae':
        # Deep (3-layer GCN) Graph Autoencoder
        model = DeepGCNModelAE(param, placeholders, num_features, features_nonzero)
    elif param['model'] == 'deep_gcn_vae':
        # Deep (3-layer GCN) Graph Variational Autoencoder
        model = DeepGCNModelVAE(param, placeholders, num_features, num_nodes,
                                features_nonzero)
    else:
        raise ValueError('Undefined model!')

    # Optimizer
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    with tf.name_scope('optimizer'):
        # Optimizer for Non-Variational Autoencoders
        if param['model'] in ('gcn_ae', 'linear_ae', 'deep_gcn_ae'):
            opt = OptimizerAE(params = param,
                              preds = model.reconstructions,
                              labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                            validate_indices = False), [-1]),
                              pos_weight = pos_weight,
                              norm = norm)
        # Optimizer for Variational Autoencoders
        elif param['model'] in ('gcn_vae', 'linear_vae', 'deep_gcn_vae'):
            opt = OptimizerVAE(params = param,
                               preds = model.reconstructions,
                               labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                             validate_indices = False), [-1]),
                               model = model,
                               num_nodes = num_nodes,
                               pos_weight = pos_weight,
                               norm = norm)

    # Normalization and preprocessing on adjacency matrix
    adj_norm = preprocess_graph(adj)
    adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))

    # Initialize TF session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Model training
    if param['verbose']:
        print("Training...")

    for epoch in range(param['epochs']):
        # Flag to compute running time for each epoch
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features,
                                        placeholders)
        feed_dict.update({placeholders['dropout']: param['dropout']})
        # Weights update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                        feed_dict = feed_dict)
        # Compute average loss
        avg_cost = outs[1]
        if param['verbose']:
            # Display epoch information
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "time=", "{:.5f}".format(time.time() - t))
            # Validation, for Link Prediction
            if not param['kcore'] and param['validation'] and param['task'] == 'link_prediction':
                feed_dict.update({placeholders['dropout']: 0})
                emb = sess.run(model.z_mean, feed_dict = feed_dict)
                feed_dict.update({placeholders['dropout']: param['dropout']})
                val_roc, val_ap = get_roc_score(valEdges, valEdgesFalse, emb)
                print("val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap))

    # Flag to compute Graph AE/VAE training time
    t_model = time.time()


    # Compute embedding

    # Get embedding from model
    emb = sess.run(model.z_mean, feed_dict = feed_dict)

    # If k-core is used, only part of the nodes from the original
    # graph are embedded. The remaining ones are projected in the
    # latent space via the expand_embedding heuristic
    if param['kcore']:
        if param['verbose']:
            print("Propagation to remaining nodes...")
        # Project remaining nodes in latent space
        emb = expandEmbedding(adj_orig, emb, nodes_kcore, param['nb_iterations'])
        # Compute mean running times for K-Core, GAE Train and Propagation steps
        mean_time_expand.append(time.time() - t_model)
        mean_time_train.append(t_model - t_core)
        mean_time_kcore.append(t_core - t_start)
        # Compute mean size of K-Core graph
        # Note: size is fixed if task is node clustering, but will vary if
        # task is link prediction due to edge masking
        mean_core_size.append(len(nodes_kcore))

    # Compute mean total running time
    mean_time.append(time.time() - t_start)


    # Test model
    if param['verbose']:
        print("Testing model...")
    # Link Prediction: classification edges/non-edges
    if param['task'] == 'link_prediction':
        # Get ROC and AP scores
        roc_score, ap_score = get_roc_score(testEdges, testEdgesFalse, emb)
        # Report scores
        mean_roc.append(roc_score)
        mean_ap.append(ap_score)

    # Node Clustering: K-Means clustering in embedding space
    elif param['task'] == 'node_clustering':
        # Clustering in embedding space
        mi_score = clustering_latent_space(emb, labels)
        # Report Adjusted Mutual Information (AMI)
        mean_mutual_info.append(mi_score)


###### Report Final Results ######

# Report final results
print("\nTest results for", param['model'], "model on", param['dataset'], "on", param['task'], "\n",
      "___________________________________________________\n")

if param['task'] == 'link_prediction':
    print("AUC scores\n", mean_roc)
    print("Mean AUC score: ", np.mean(mean_roc), "\nStd of AUC scores: ", np.std(mean_roc), "\n \n")

    print("AP scores\n", mean_ap)
    print("Mean AP score: ", np.mean(mean_ap), "\nStd of AP scores: ", np.std(mean_ap), "\n \n")

else:
    print("Adjusted MI scores\n", mean_mutual_info)
    print("Mean Adjusted MI score: ", np.mean(mean_mutual_info), "\nStd of Adjusted MI scores: ", np.std(mean_mutual_info), "\n \n")

print("Total Running times\n", mean_time)
print("Mean total running time: ", np.mean(mean_time), "\nStd of total running time: ", np.std(mean_time), "\n \n")

if param['kcore']:
    print("Details on degeneracy framework, with k =", param['k'], ": \n \n")

    print("Running times for k-core decomposition\n", mean_time_kcore)
    print("Mean: ", np.mean(mean_time_kcore), "\nStd: ", np.std(mean_time_kcore), "\n \n")

    print("Running times for autoencoder training\n", mean_time_train)
    print("Mean: ", np.mean(mean_time_train), "\nStd: ", np.std(mean_time_train), "\n \n")

    print("Running times for propagation\n", mean_time_expand)
    print("Mean: ", np.mean(mean_time_expand), "\nStd: ", np.std(mean_time_expand), "\n \n")

    print("Sizes of k-core subgraph\n", mean_core_size)
    print("Mean: ", np.mean(mean_core_size), "\nStd: ", np.std(mean_core_size), "\n \n")