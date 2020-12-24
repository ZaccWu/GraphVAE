from __future__ import division
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score, adjusted_mutual_info_score
import numpy as np
import tensorflow as tf

def sigmoid(x):
    """ Sigmoid activation function
    :param x: scalar value
    :return: sigmoid activation
    """
    return 1 / (1 + np.exp(-x))

def getRocScore(edgesPos, edgesNeg, emb):
    """ Link Prediction: computes AUC ROC and AP scores from embeddings vectors,
    and from ground-truth lists of positive and negative node pairs
    :param edgesPos: list of positive node pairs
    :param edgesNeg: list of negative node pairs
    :param emb: n*d matrix of embedding vectors for all graph nodes
    :return: Area Under ROC Curve (AUC ROC) and Average Precision (AP) scores
    """
    preds = []
    predsNeg = []
    for e in edgesPos:
        # Link Prediction on positive pairs
        preds.append(sigmoid(emb[e[0],:].dot(emb[e[1],:].T)))
    for e in edgesNeg:
        # Link Prediction on negative pairs
        predsNeg.append(sigmoid(emb[e[0],:].dot(emb[e[1],:].T)))

    # Stack all predictions and labels
    predsAll = np.hstack([preds, predsNeg])
    labelsAll = np.hstack([np.ones(len(preds)), np.zeros(len(predsNeg))])

    # Computes metrics
    rocScore = roc_auc_score(labelsAll, predsAll)
    apScore = average_precision_score(labelsAll, predsAll)
    return rocScore, apScore

def clusteringLatentSpace(emb, label, nbClusters=None):
    """ Node Clustering: computes Adjusted Mutual Information score from a
    K-Means clustering of nodes in latent embedding space
    :param emb: n*d matrix of embedding vectors for all graph nodes
    :param label: ground-truth node labels
    :param nbClusters: int number of ground-truth communities in graph
    :return: Adjusted Mutual Information (AMI) score
    """
    if nbClusters is None:
        nbClusters = len(np.unique(label))
    # K-Means Clustering
    clusteringPred = KMeans(n_clusters = nbClusters, init ='k-means++').fit(emb).labels_
    # Compute metrics
    return adjusted_mutual_info_score(label, clusteringPred)