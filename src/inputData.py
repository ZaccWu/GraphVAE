import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys

'''
Disclaimer: the functions from this file come from tkipf/gae
original repository on Graph Autoencoders
'''

def parseIndexFile(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def fixCiteseerDataX(x, tx, testIdxReorder, testIdxRange):
    # Fix citeseer dataset (there are some isolated nodes in the graph)
    # Find isolated nodes, add them as zero-vecs into the right position
    testIdxRangeFull = range(min(testIdxReorder), max(testIdxReorder) + 1)
    txExtended = sp.lil_matrix((len(testIdxRangeFull), x.shape[1]))
    txExtended[testIdxRange - min(testIdxRange), :] = tx
    tx = txExtended
    return tx

def fixCiteseerDataY(ty, testIdxReorder, testIdxRange):
    # Fix citeseer dataset (there are some isolated nodes in the graph)
    # Find isolated nodes, add them as zero-vecs into the right position
    testIdxRangeFull = range(min(testIdxReorder), max(testIdxReorder) + 1)
    tyExtended = np.zeros((len(testIdxRangeFull), ty.shape[1]))
    tyExtended[testIdxRange - min(testIdxRange), :] = ty
    ty = tyExtended
    return ty

def openDataFiles(dataset, names):
    objects = []
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    return objects

def loadData(dataset):
    """ Load datasets from tkipf/gae input files
    :param dataset: 'cora', 'citeseer' or 'pubmed' graph dataset.
    :return: n*n sparse adjacency matrix and n*f node features matrix
    """
    # Load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = openDataFiles(dataset, names)
    x, tx, allx, graph = tuple(objects)
    testIdxReorder = parseIndexFile("../data/ind.{}.test.index".format(dataset))
    testIdxRange = np.sort(testIdxReorder)
    if dataset == 'citeseer':
        tx = fixCiteseerDataX(x, tx, testIdxReorder, testIdxRange)

    features = sp.vstack((allx, tx)).tolil()
    features[testIdxReorder, :] = features[testIdxRange, :]
    graph = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(graph)
    return adj, features

def loadLabel(dataset):
    """ Load node-level labels from tkipf/gae input files
    :param dataset: 'cora', 'citeseer' or 'pubmed' graph dataset.
    :return: n-dim array of node labels (used for clustering)
    """
    names = ['ty', 'ally']
    objects = openDataFiles(dataset, names)
    ty, ally = tuple(objects)
    testIdxReorder = parseIndexFile("../data/ind.{}.test.index".format(dataset))
    testIdxRange = np.sort(testIdxReorder)
    if dataset == 'citeseer':
        ty = fixCiteseerDataY(ty, testIdxReorder, testIdxRange)

    label = sp.vstack((ally, ty)).tolil()
    label[testIdxReorder, :] = label[testIdxRange, :]
    # One-hot to integers
    label = np.argmax(label.toarray(), axis = 1)
    return label