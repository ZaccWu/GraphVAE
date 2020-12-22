from __future__ import division
from __future__ import print_function
from util.evaluation import getRocScore, clusteringLatentSpace
from src.inputData import loadData, loadLabel
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

meanROC = []   # link prediction
meanAP = []    # link prediction
meanMutualInfo = []   # node clustering
meanTimeKcore = []    # kcore
meanTimeTrain = []    # kcore
meanTimeExpand = []   # kcore
meanCoreSize = []     # kcore
meanTime = []  # general

# Load graph dataset
if param['verbose']:
    print("Loading data...")
adjInit, featuresInit = loadData(param['dataset'])
# Load ground-truth labels for node clustering task
if param['task'] == 'node_clustering':
    labels = loadLabel(param['dataset'])

###### Run the Experiments ######
# The entire training+test process is repeated nb_run times
for i in range(param['nb_run']):
    if param['task'] == 'link_prediction' :
        if param['verbose']:
            print("Masking test edges...")
        # Edge Masking for Link Prediction: compute Train/Validation/Test set
        adj, valEdges, valEdgesFalse, testEdges, testEdgesFalse = maskTestEdges(adjInit, param['prop_test'], param['prop_val'])
    elif param['task'] == 'node_clustering':
        adjTri = sp.triu(adjInit)
        adj = adjTri + adjTri.T
    else:
        raise ValueError('Undefined task!')

    # Start computation of running times
    tStart = time.time()

    # Degeneracy Framework / K-Core Decomposition
    if param['kcore']:
        if param['verbose']:
            print("Starting k-core decomposition of the graph")
        # Save adjacency matrix of un-decomposed graph
        # (needed to embed nodes that are not in k-core, after GAE training)
        adjOrig = adj
        # Get the (smaller) adjacency matrix of the k-core subgraph,
        # and the corresponding nodes
        adj, nodesKcore = computeKcore(adj, param['k'])
        # Get the (smaller) feature matrix of the nb_core graph
        if param['features']:
            features = featuresInit[nodesKcore, :]
        # Flag to compute k-core decomposition's running time
        tCore = time.time()
    elif param['features']:
        features = featuresInit

    # Preprocessing and initialization
    if param['verbose']:
        print("Preprocessing and Initializing...")
    # Compute number of nodes
    numNodes = adj.shape[0]
    # If features are not used, replace feature matrix by identity matrix
    if not param['features']:
        features = sp.identity(adj.shape[0])
    # Preprocessing on node features
    features = sparseToTuple(features)
    numFeatures = features[2][1]
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
        model = GCNModelAE(param, placeholders, numFeatures, features_nonzero)
    elif param['model'] == 'gcn_vae':
        # Standard Graph Variational Autoencoder
        model = GCNModelVAE(param, placeholders, numFeatures, numNodes,
                            features_nonzero)
    elif param['model'] == 'linear_ae':
        # Linear Graph Autoencoder
        model = LinearModelAE(param, placeholders, numFeatures, features_nonzero)
    elif param['model'] == 'linear_vae':
        # Linear Graph Variational Autoencoder
        model = LinearModelVAE(param, placeholders, numFeatures, numNodes,
                               features_nonzero)
    elif param['model'] == 'deep_gcn_ae':
        # Deep (3-layer GCN) Graph Autoencoder
        model = DeepGCNModelAE(param, placeholders, numFeatures, features_nonzero)
    elif param['model'] == 'deep_gcn_vae':
        # Deep (3-layer GCN) Graph Variational Autoencoder
        model = DeepGCNModelVAE(param, placeholders, numFeatures, numNodes,
                                features_nonzero)
    else:
        raise ValueError('Undefined model!')

    # Optimizer
    posWeight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    with tf.name_scope('optimizer'):
        # Optimizer for Non-Variational Autoencoders
        if param['model'] in ('gcn_ae', 'linear_ae', 'deep_gcn_ae'):
            opt = OptimizerAE(params = param,
                              preds = model.reconstructions,
                              labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                            validate_indices = False), [-1]),
                              pos_weight = posWeight,
                              norm = norm)
        # Optimizer for Variational Autoencoders
        elif param['model'] in ('gcn_vae', 'linear_vae', 'deep_gcn_vae'):
            opt = OptimizerVAE(params = param,
                               preds = model.reconstructions,
                               labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                             validate_indices = False), [-1]),
                               model = model,
                               num_nodes = numNodes,
                               pos_weight = posWeight,
                               norm = norm)

    # Normalization and preprocessing on adjacency matrix
    adjNorm = preprocessGraph(adj)
    adjLabel = sparseToTuple(adj + sp.eye(adj.shape[0]))

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
        feedDict = constructFeedDict(adjNorm, adjLabel, features,
                                     placeholders)
        feedDict.update({placeholders['dropout']: param['dropout']})
        # Weights update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                        feed_dict = feedDict)
        # Compute average loss
        avg_cost = outs[1]
        if param['verbose']:
            # Display epoch information
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "time=", "{:.5f}".format(time.time() - t))
            # Validation, for Link Prediction
            if not param['kcore'] and param['validation'] and param['task'] == 'link_prediction':
                feedDict.update({placeholders['dropout']: 0})
                emb = sess.run(model.z_mean, feed_dict = feedDict)
                feedDict.update({placeholders['dropout']: param['dropout']})
                val_roc, val_ap = getRocScore(valEdges, valEdgesFalse, emb)
                print("val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap))

    # Flag to compute Graph AE/VAE training time
    tModel = time.time()


    # Compute embedding

    # Get embedding from model
    emb = sess.run(model.z_mean, feed_dict = feedDict)

    # If k-core is used, only part of the nodes from the original
    # graph are embedded. The remaining ones are projected in the
    # latent space via the expand_embedding heuristic
    if param['kcore']:
        if param['verbose']:
            print("Propagation to remaining nodes...")
        # Project remaining nodes in latent space
        emb = expandEmbedding(adjOrig, emb, nodesKcore, param['nb_iterations'])
        # Compute mean running times for K-Core, GAE Train and Propagation steps
        meanTimeExpand.append(time.time() - tModel)
        meanTimeTrain.append(tModel - tCore)
        meanTimeKcore.append(tCore - tStart)
        # Compute mean size of K-Core graph
        # Note: size is fixed if task is node clustering, but will vary if
        # task is link prediction due to edge masking
        meanCoreSize.append(len(nodesKcore))

    # Compute mean total running time
    meanTime.append(time.time() - tStart)


    # Test model
    if param['verbose']:
        print("Testing model...")
    # Link Prediction: classification edges/non-edges
    if param['task'] == 'link_prediction':
        # Get ROC and AP scores
        rocScore, apScore = getRocScore(testEdges, testEdgesFalse, emb)
        # Report scores
        meanROC.append(rocScore)
        meanAP.append(apScore)

    # Node Clustering: K-Means clustering in embedding space
    elif param['task'] == 'node_clustering':
        # Clustering in embedding space
        miScore = clusteringLatentSpace(emb, labels)
        # Report Adjusted Mutual Information (AMI)
        meanMutualInfo.append(miScore)


###### Report Final Results ######

# Report final results
print("\nTest results for", param['model'], "model on", param['dataset'], "on", param['task'], "\n",
      "___________________________________________________\n")

if param['task'] == 'link_prediction':
    print("AUC scores\n", meanROC)
    print("Mean AUC score: ", np.mean(meanROC), "\nStd of AUC scores: ", np.std(meanROC), "\n \n")

    print("AP scores\n", meanAP)
    print("Mean AP score: ", np.mean(meanAP), "\nStd of AP scores: ", np.std(meanAP), "\n \n")

else:
    print("Adjusted MI scores\n", meanMutualInfo)
    print("Mean Adjusted MI score: ", np.mean(meanMutualInfo), "\nStd of Adjusted MI scores: ", np.std(meanMutualInfo), "\n \n")

print("Total Running times\n", meanTime)
print("Mean total running time: ", np.mean(meanTime), "\nStd of total running time: ", np.std(meanTime), "\n \n")

if param['kcore']:
    print("Details on degeneracy framework, with k =", param['k'], ": \n \n")

    print("Running times for k-core decomposition\n", meanTimeKcore)
    print("Mean: ", np.mean(meanTimeKcore), "\nStd: ", np.std(meanTimeKcore), "\n \n")

    print("Running times for autoencoder training\n", meanTimeTrain)
    print("Mean: ", np.mean(meanTimeTrain), "\nStd: ", np.std(meanTimeTrain), "\n \n")

    print("Running times for propagation\n", meanTimeExpand)
    print("Mean: ", np.mean(meanTimeExpand), "\nStd: ", np.std(meanTimeExpand), "\n \n")

    print("Sizes of k-core subgraph\n", meanCoreSize)
    print("Mean: ", np.mean(meanCoreSize), "\nStd: ", np.std(meanCoreSize), "\n \n")