import scipy.io as scio
import os.path as osp

import numpy as np


from sklearn.preprocessing import OneHotEncoder

import torch


from numpy import linalg as LA

def mat_reader(mat_path, attribute: int):

    if attribute < 0 or attribute > 7:
        print("参数仅能在0-7间选择，0-7分别代表__header__, __version__, __globals__, label, fmri, dti, X_ori, X_normalize")
        return

    BASEDIR = osp.dirname(osp.abspath(__file__))

    path_mat = osp.join(BASEDIR, mat_path)
    data = scio.loadmat(path_mat)
    return_group = []
    for i, k in enumerate(data):
        if i == attribute:
            return_group.append(data[k])
    return return_group




# for degree_bin node features
def binning(a, n_bins=10):
    n_graphs = a.shape[0]
    n_nodes = a.shape[1]
    _, bins = np.histogram(a, n_bins)
    binned = np.digitize(a, bins)
    binned = binned.reshape(-1, 1)
    enc = OneHotEncoder()
    return enc.fit_transform(binned).toarray().reshape(n_graphs, n_nodes, -1).astype(np.float32)


# for LDP node features
def LDP(g, key='deg'):
    x = np.zeros([len(g.nodes()), 5])

    deg_dict = dict(nx.degree(g))
    for n in g.nodes():
        g.nodes[n][key] = deg_dict[n]

    for i in g.nodes():
        nodes = g[i].keys()

        nbrs_deg = [g.nodes[j][key] for j in nodes]

        if len(nbrs_deg) != 0:
            x[i] = [
                np.mean(nbrs_deg),
                np.min(nbrs_deg),
                np.max(nbrs_deg),
                np.std(nbrs_deg),
                np.sum(nbrs_deg)
            ]

    return x

def compute_x(a1, node_features):
    # construct node features X
    if node_features == 'identity':
        x = torch.cat([torch.diag(torch.ones(a1.shape[1]))] * a1.shape[0]).reshape([a1.shape[0], a1.shape[1], -1])
        x1 = x.clone()

    elif node_features == 'node2vec':
        X = np.load(f'./{dataset_name}_{modality}.emb', allow_pickle=True).astype(np.float32)
        x1 = torch.from_numpy(X)

    elif node_features == 'degree':
        a1b = (a1 != 0).float()
        x1 = a1b.sum(dim=2, keepdim=True)

    elif node_features == 'degree_bin':
        a1b = (a1 != 0).float()
        x1 = binning(a1b.sum(dim=2))

    elif node_features == 'adj': # edge profile
        x1 = a1.float()

    elif node_features == 'LDP': # degree profile
        a1b = (a1 != 0).float()
        x1 = []
        n_graphs: int = a1.shape[0]
        for i in range(n_graphs):
            x1.append(LDP(nx.from_numpy_array(a1b[i].numpy())))

    elif node_features == 'eigen':
        _, x = LA.eig(a1.numpy())

    x1 = torch.Tensor(x1).float()
    return x1

def compute_x_wzt_changed(adj_matrix, node_features):

    w = adj_matrix

    # construct node features X
    if node_features == 'identity':
        x = torch.cat([torch.diag(torch.ones(w.shape[1]))] * w.shape[0]).reshape([w.shape[0], w.shape[1], -1])
        x1 = x.clone()

    elif node_features == 'degree':
        a1b = torch.from_numpy((w != 0).astype(np.float32))
        x1 = a1b.sum(dim=2, keepdim=True)

    elif node_features == 'degree_bin':
        a1b = torch.from_numpy((w != 0).astype(np.float32))
        x1 = binning(a1b.sum(dim=2))

    elif node_features == 'adj': # edge profile
        x1 = torch.from_numpy(w).float()

    elif node_features == 'LDP': # degree profile
        a1b = torch.from_numpy((w != 0).astype(np.float32))
        x1 = []
        n_graphs: int = w.shape[0]
        for i in range(n_graphs):
            x1.append(LDP(nx.from_numpy_array(w[i])))
        x1 = torch.Tensor(x1).float()

    elif node_features == 'eigen':
        _, x1 = LA.eig(w)

    return x1

import networkx as nx

def convert_to_networkx(x):
    graphs = []
    for k in range(x.shape[-1]): # 遍历所有特征维度
        g = nx.Graph()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                g.add_node((i, j), feature=x[i, j, k])
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if i > 0:
                    g.add_edge((i, j), (i - 1, j))
                if j > 0:
                    g.add_edge((i, j), (i, j - 1))
        graphs.append(g)
    return graphs



def convert_to_networkx2(adj_matrix,feat_tensor,node_labels):
    graphs = []


    for inum_graph in range(feat_tensor.shape[0]):
        G = nx.from_numpy_matrix(adj_matrix[inum_graph])
        G.graph['label'] = node_labels[inum_graph]
        for i in range(feat_tensor.shape[1]):
            for j in range(feat_tensor.shape[2]):
                G.nodes[i][j] = feat_tensor[inum_graph,i, j ].tolist()

        graphs.append(G)
    return graphs

