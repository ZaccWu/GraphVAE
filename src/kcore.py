import networkx as nx
import numpy as np
import scipy.sparse as sp
import warnings

warnings.simplefilter('ignore', sp.SparseEfficiencyWarning)

def computeKcore(adj, nbCore):
    """ Computes the k-core version of a graph - See IJCAI 2019 paper
    for theoretical details on k-core decomposition
    :param adj: sparse adjacency matrix of the graph
    :param nbCore: a core number, from 0 to the "degeneracy"
                    (i.e. max core value) of the graph
    :return: sparse adjacency matrix of the nb_core-core subgraph, together
             with the list of nodes from this core
    """
    # Preprocessing on graph
    G = nx.from_scipy_sparse_matrix(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    # K-core decomposition
    coreNumber = nx.core_number(G)
    # nb_core subgraph
    kcore = nx.k_core(G, nbCore, coreNumber)
    # Get list of nodes from this subgraph
    nodesKcore = kcore.nodes
    # Adjacency matrix of this subgraph
    adjKcore = nx.adjacency_matrix(kcore)
    return adjKcore, nodesKcore

def expandEmbedding(adj, embKcore, nodesKcore, nbIterations):
    """ Algorithm 2 'Propagation of latent representation' from IJCAI 2019 paper
    Propagates embeddings vectors computed on k-core to the remaining nodes
    of the graph (i.e. the nodes outside of the k-core)
    :param adj: sparse adjacency matrix of the graph
    :param embKcore: n*d embedding matrix computed from Graph AE/VAE
                      for nodes in k-core
    :param nodesKcore: list of nodes in k-core
    :param nbIterations: number of iterations "t" for fix-point iteration
                          strategy of Algorithm 2
    :return: n*d matrix of d-dim embeddings for all nodes of the graph
    """
    # Initialization
    numNodes = adj.shape[0]
    emb = sp.csr_matrix((numNodes, embKcore.shape[1]))
    emb[nodesKcore, :] = embKcore
    adj = adj.tocsr()
    embeddedNodes = []
    newEmbeddedNodes = np.array(nodesKcore)

    # Assign latent space representation to nodes that were not in k-core
    while len(newEmbeddedNodes) > 0:
        embeddedNodes = np.hstack((embeddedNodes, newEmbeddedNodes))
        # Get nodes from V2 set
        reachedNodes = np.setdiff1d(np.where((adj[newEmbeddedNodes,:].sum(0) != 0)), embeddedNodes)
        # Nodes from V1 (newly embedded) and V2
        newEmbeddedNodesUnionReached = np.union1d(newEmbeddedNodes, reachedNodes)
        # Adjacency matrices normalization by total degree in (A1,A2)
        adj12 = adj[reachedNodes,:][:,newEmbeddedNodesUnionReached]
        degrees = np.array(adj12.sum(1))
        degreeMat = sp.diags(np.power(degrees, -1.0).flatten())
        adj1 = degreeMat.dot(adj[reachedNodes,:][:,newEmbeddedNodes])
        adj2 = degreeMat.dot(adj[reachedNodes,:][:,reachedNodes])

        # Iterations
        z1 = emb[newEmbeddedNodes,:]
        z2 = np.random.random_sample((len(reachedNodes), emb.shape[1]))
        for j in range(nbIterations):
            z2 = adj1.dot(z1) + adj2.dot(z2)
        emb[reachedNodes,:] += z2
        # Update new_embedded_nodes
        newEmbeddedNodes = reachedNodes

    # Handle isolated nodes
    emb[emb.getnnz(1) == 0] = np.mean(embKcore, axis=0)
    # Return embedding
    return emb.toarray()