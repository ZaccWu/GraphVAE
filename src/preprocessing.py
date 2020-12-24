import numpy as np
import scipy.sparse as sp
import tensorflow as tf

def sparseToTuple(sparseMx):
    if not sp.isspmatrix_coo(sparseMx):
        sparseMx = sparseMx.tocoo()
    coords = np.vstack((sparseMx.row, sparseMx.col)).transpose()
    values = sparseMx.data
    shape = sparseMx.shape
    return coords, values, shape

def preprocessGraph(adj):
    adj = sp.coo_matrix(adj)
    adjDot = adj + sp.eye(adj.shape[0])
    degreeMatInvSqrt = sp.diags(np.power(np.array(adjDot.sum(1)), -0.5).flatten())
    adjNormalized = adjDot.dot(degreeMatInvSqrt).transpose().dot(degreeMatInvSqrt)
    return sparseToTuple(adjNormalized)

def constructFeedDict(adjNormalized, adj, features, placeholders):
    # Construct feed dictionary
    feedDict = dict()
    feedDict.update({placeholders['features']: features})
    feedDict.update({placeholders['adj']: adjNormalized})
    feedDict.update({placeholders['adj_orig']: adj})
    return feedDict

def maskTestEdges(adj, testPercent=10., valPercent=5.):
    """ Randomly removes some edges from original graph to create
    test and validation sets for link prediction task
    :param adj: complete sparse adjacency matrix of the graph
    :param testPercent: percentage of edges in test set
    :param valPercent: percentage of edges in validation set
    :return: train incomplete adjacency matrix, validation and test sets
    """
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[None, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert adj.diagonal().sum() == 0

    edgesPositive, _, _ = sparseToTuple(adj)
    # Filtering out edges from lower triangle of adjacency matrix
    edgesPositive = edgesPositive[edgesPositive[:,1] > edgesPositive[:,0],:]
    # val_edges, val_edges_false, test_edges, test_edges_false = None, None, None, None

    # number of positive (and negative) edges in test and val sets:
    numTest = int(np.floor(edgesPositive.shape[0] / (100. / testPercent)))
    numVal = int(np.floor(edgesPositive.shape[0] / (100. / valPercent)))

    # sample positive edges for test and val sets:
    edgesPositiveIdx = np.arange(edgesPositive.shape[0])
    np.random.shuffle(edgesPositiveIdx)
    valEdgeIdx = edgesPositiveIdx[:numVal]
    testEdgeIdx = edgesPositiveIdx[numVal:(numVal + numTest)]
    testEdges = edgesPositive[testEdgeIdx] # positive test edges
    valEdges = edgesPositive[valEdgeIdx] # positive val edges
    trainEdges = np.delete(edgesPositive, np.hstack([testEdgeIdx, valEdgeIdx]), axis = 0) # positive train edges

    # the above strategy for sampling without replacement will not work for
    # sampling negative edges on large graphs, because the pool of negative
    # edges is much much larger due to sparsity, therefore we'll use
    # the following strategy:
    # 1. sample random linear indices from adjacency matrix WITH REPLACEMENT
    # (without replacement is super slow). sample more than we need so we'll
    # probably have enough after all the filtering steps.
    # 2. remove any edges that have already been added to the other edge lists
    # 3. convert to (i,j) coordinates
    # 4. swap i and j where i > j, to ensure they're upper triangle elements
    # 5. remove any duplicate elements if there are any
    # 6. remove any diagonal elements
    # 7. if we don't have enough edges, repeat this process until we get enough
    positiveIdx, _, _ = sparseToTuple(adj) # [i,j] coord pairs for all true edges
    positiveIdx = positiveIdx[:,0]*adj.shape[0] + positiveIdx[:,1] # linear indices
    testEdgesFalse = np.empty((0,2),dtype='int64')
    idxTestEdgesFalse = np.empty((0,),dtype='int64')

    while len(testEdgesFalse) < len(testEdges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(numTest - len(testEdgesFalse)), replace = True)
        # step 2:
        idx = idx[~np.in1d(idx, positiveIdx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idxTestEdgesFalse, assume_unique = True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx,colidx)).transpose()
        # step 4:
        lowertrimask = coords[:,0] > coords[:,1]
        coords[lowertrimask] = coords[lowertrimask][:,::-1]
        # step 5:
        coords = np.unique(coords, axis = 0) # note: coords are now sorted lexicographically
        np.random.shuffle(coords) # not anymore
        # step 6:
        coords = coords[coords[:,0] != coords[:,1]]
        # step 7:
        coords = coords[:min(numTest, len(idx))]
        testEdgesFalse = np.append(testEdgesFalse, coords, axis = 0)
        idx = idx[:min(numTest, len(idx))]
        idxTestEdgesFalse = np.append(idxTestEdgesFalse, idx)

    valEdgesFalse = np.empty((0,2), dtype = 'int64')
    idxValEdgesFalse = np.empty((0,), dtype = 'int64')
    while len(valEdgesFalse) < len(valEdges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(numVal - len(valEdgesFalse)), replace = True)
        # step 2:
        idx = idx[~np.in1d(idx, positiveIdx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idxTestEdgesFalse, assume_unique = True)]
        idx = idx[~np.in1d(idx, idxValEdgesFalse, assume_unique = True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx,colidx)).transpose()
        # step 4:
        lowertrimask = coords[:,0] > coords[:,1]
        coords[lowertrimask] = coords[lowertrimask][:,::-1]
        # step 5:
        coords = np.unique(coords, axis = 0) # note: coords are now sorted lexicographically
        np.random.shuffle(coords) # not any more
        # step 6:
        coords = coords[coords[:,0] != coords[:,1]]
        # step 7:
        coords = coords[:min(numVal, len(idx))]
        valEdgesFalse = np.append(valEdgesFalse, coords, axis = 0)
        idx = idx[:min(numVal, len(idx))]
        idxValEdgesFalse = np.append(idxValEdgesFalse, idx)

    # sanity checks:
    trainEdgesLinear = trainEdges[:,0]*adj.shape[0] + trainEdges[:,1]
    testEdgesLinear = testEdges[:,0]*adj.shape[0] + testEdges[:,1]
    assert not np.any(np.in1d(idxTestEdgesFalse, positiveIdx))
    assert not np.any(np.in1d(idxValEdgesFalse, positiveIdx))
    assert not np.any(np.in1d(valEdges[:,0]*adj.shape[0]+valEdges[:,1], trainEdgesLinear))
    assert not np.any(np.in1d(testEdgesLinear, trainEdgesLinear))
    assert not np.any(np.in1d(valEdges[:,0]*adj.shape[0]+valEdges[:,1], testEdgesLinear))

    # Re-build adj matrix
    data = np.ones(trainEdges.shape[0])
    adjTrain = sp.csr_matrix((data, (trainEdges[:, 0], trainEdges[:, 1])), shape=adj.shape)
    adjTrain = adjTrain + adjTrain.T
    return adjTrain, valEdges, valEdgesFalse, testEdges, testEdgesFalse