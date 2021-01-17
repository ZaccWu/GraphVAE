from __future__ import division
from __future__ import print_function
from util.evaluation import getRocScore, clusteringLatentSpace
from src.inputData import loadData, loadLabel
from src.kcore import computeKcore, expandEmbedding
from src.models import gcnAE, gcnVAE, linearAE, linearVAE, gcnDeepAE, gcnDeepVAE, gcnMeanVAE, gcnStdVAE
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
    # select the model
    'model': 'gcn_vae',              # 'gcn_ae', 'gcn_vae', 'linear_ae', 'linear_vae', 'deep_gcn_ae', 'deep_gcn_vae', 'gcn_mean_vae', 'gcn_std_vae'
    # model parameters
    'dropout': 0.,                  # Dropout rate (1 - keep probability)
    'epochs': 200,
    'features': True,
    'learning_rate': 0.01,
    'hidden': 32,                   # Number of units in GCN hidden layer(s)
    'dimension': 16,                # Embedding dimension (Dimension of encoder output)
    # experimental parameters
    'nb_run':5,                    # Number of model run + test
    'prop_val': 5.,                 # Proportion of edges in validation set (link prediction)
    'prop_test': 10.,               # Proportion of edges in test set (link prediction)
    'validation': False,            # Whether to report validation results at each epoch (link prediction)
    'verbose': True,                # Whether to print comments details
    # degeneracy framework parameters
    'kcore': False,                 # Whether to run k-core decomposition (False-train on the entire graph)
    'k': 2,
    'nb_iterations': 10,
    # betaVAE
    'beta': 1,
}

# Lists to collect average results
meanROC = []   # link prediction
meanAP = []    # link prediction
meanTime = []  # general

# Load graph dataset
if param['verbose']:
    print("Loading data...")
adjInit, featuresInit = loadData(param['dataset'])

'''
Data Description
adjInit <class 'scipy.sparse.csr.csr_matrix'>
Cora: (2708,2708)
Citeseer: (3327,3327)
Pubmed: (19717,19717)

featuresInit <class 'scipy.sparse.lil.lil_matrix'>
Cora: (2708,1433)
Citeseer: (3327,3703)
Pubmed: (19717,500)
'''

###### Run the Experiments ######
# The entire training+test process is repeated nb_run times
for i in range(param['nb_run']):
    if param['verbose']:
        print("Masking test edges...")
    # Edge Masking for Link Prediction: compute Train/Validation/Test set
    adj, valEdges, valEdgesFalse, testEdges, testEdgesFalse = maskTestEdges(adjInit, param['prop_test'], param['prop_val'])
    tStart = time.time()

    # Degeneracy Framework / K-Core Decomposition
    if param['kcore']:
        if param['verbose']:
            print("Starting k-core decomposition of the graph")
        # Save adjacency matrix of un-decomposed graph (needed to embed nodes that are not in k-core, after GAE training)
        adjCopy = adj
        # Get the (smaller) adjacency matrix of the k-core subgraph, and the corresponding nodes
        adj, nodesKcore = computeKcore(adj, param['k'])
        # Get the (smaller) feature matrix of the nb_core graph
        if param['features']:
            features = featuresInit[nodesKcore, :]
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
    features = sparseToTuple(features)  # features: tuple
    numFeatures = features[2][1]
    features_nonzero = features[1].shape[0]

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape = ())
    }

    ModelDict = {
        'gcn_ae': gcnAE(param, placeholders, numFeatures, features_nonzero),
        'gcn_vae': gcnVAE(param, placeholders, numFeatures, numNodes, features_nonzero),
        'linear_ae': linearAE(param, placeholders, numFeatures, features_nonzero),
        'linear_vae': linearVAE(param, placeholders, numFeatures, numNodes, features_nonzero),
        'deep_gcn_ae': gcnDeepAE(param, placeholders, numFeatures, features_nonzero),
        'deep_gcn_vae': gcnDeepVAE(param, placeholders, numFeatures, numNodes, features_nonzero),
        'gcn_mean_vae': gcnMeanVAE(param, placeholders, numFeatures, numNodes, features_nonzero),
        'gcn_std_vae': gcnStdVAE(param, placeholders, numFeatures, numNodes, features_nonzero),
    }
    model = ModelDict.get(param['model'])
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
                              posWeight= posWeight,
                              norm = norm)
        # Optimizer for Variational Autoencoders
        elif param['model'] in ('gcn_vae', 'linear_vae', 'deep_gcn_vae', 'gcn_mean_vae', 'gcn_std_vae'):
            opt = OptimizerVAE(params = param,
                               preds = model.reconstructions,
                               labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                             validate_indices = False), [-1]),
                               model = model,
                               numNodes= numNodes,
                               posWeight= posWeight,
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
        # Construct feed dictionary
        feedDict = constructFeedDict(adjNorm, adjLabel, features, placeholders)
        feedDict.update({placeholders['dropout']: param['dropout']})
        # Weights update
        outs = sess.run([opt.optOp, opt.cost, opt.accuracy], feed_dict = feedDict)
        # Compute average loss
        avgCost = outs[1]
        if param['verbose']:
            # Display epoch information
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avgCost))
            # Validation, for Link Prediction
            if not param['kcore'] and param['validation']:
                feedDict.update({placeholders['dropout']: 0})
                emb = sess.run(model.zMean, feed_dict = feedDict)
                feedDict.update({placeholders['dropout']: param['dropout']})
                val_roc, val_ap = getRocScore(valEdges, valEdgesFalse, emb)
                print("val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap))

    # Compute embedding
    # Get embedding from model
    emb = sess.run(model.zMean, feed_dict = feedDict)

    # If k-core is used, only part of the nodes from the original graph are embedded.
    # The remaining ones are projected in the latent space via the expand_embedding heuristic
    if param['kcore']:
        if param['verbose']:
            print("Propagation to remaining nodes...")
        # Project remaining nodes in latent space
        emb = expandEmbedding(adjCopy, emb, nodesKcore, param['nb_iterations'])

    # Compute mean total running time
    meanTime.append(time.time() - tStart)

    '''
    # print the reconstruction/KL loss and KL for single z_j
    printll = tf.Print(opt.logLik, [opt.logLik])
    print(sess.run(printll, feed_dict = feedDict))
    printkl = tf.Print(opt.kl, [opt.kl])
    print(sess.run(printkl, feed_dict = feedDict))
    printz = tf.Print(model.z, [model.z, model.z.shape])
    allData = sess.run(printz, feed_dict = feedDict)
    for i in range(16):
        mu = np.mean(allData[i])
        sigma = np.std(allData[i])
        print(-0.5*(1+2*np.log(sigma)-sigma**2-mu**2))    
    '''

    # Test model
    if param['verbose']:
        print("Testing model...")
    # Link Prediction: classification edges/non-edges
    # Get ROC and AP scores
    rocScore, apScore = getRocScore(testEdges, testEdgesFalse, emb)
    meanROC.append(rocScore)
    meanAP.append(apScore)


###### Report Final Results ######
print("\nTest results for", param['model'], "model on", param['dataset'], "on", "link prediction", "\n",
      "___________________________________________________\n")
print("AUC scores\n", meanROC)
print("Mean AUC score: ", np.mean(meanROC), "\nStd of AUC scores: ", np.std(meanROC), "\n \n")
print("AP scores\n", meanAP)
print("Mean AP score: ", np.mean(meanAP), "\nStd of AP scores: ", np.std(meanAP), "\n \n")
print("Total Running times\n", meanTime)
print("Mean total running time: ", np.mean(meanTime), "\nStd of total running time: ", np.std(meanTime), "\n \n")
