import networkx as nx
import numpy as np
import torch.utils.data

import util

class GraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''
    def __init__(self, G_list, labels , features='default', normalize=True, max_num_nodes=0):
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []
        self.label_all = labels

        self.deg_feat_all = []
        self.max_num_nodes = max_num_nodes

        # if features == 'default':
        #     self.feat_dim = util.node_dict(G_list[0])[0]['feat'].shape[0]
        # else:
        self.feat_dim = 0

        for G in G_list:
            adj = np.array(nx.to_numpy_matrix(G))
            # np.save('adj.npy', adj)
            # adj_zhiqian = np.load('adj.npy')
            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            self.adj_all.append(adj)
            self.len_all.append(G.number_of_nodes())
            # self.label_all.append(G.graph['label'])

            degs = np.sum(np.array(adj), 1)
            if self.max_num_nodes > G.number_of_nodes():
                degs = np.expand_dims(np.pad(degs, (0, self.max_num_nodes - G.number_of_nodes()), 'constant', constant_values=0), axis=1)
            elif self.max_num_nodes < G.number_of_nodes():
                deg_index = np.argsort(degs, axis=0)
                deg_ind = deg_index[0: G.number_of_nodes()-self.max_num_nodes]
                degs = np.delete(degs, [deg_ind], axis=0)
                degs = np.expand_dims(degs, axis=1)
            else:
                degs = np.expand_dims(degs, axis=1)
            self.deg_feat_all.append(degs)



            # if features == 'default':
            # f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
            n = len(util.node_dict(G)[0])
            # f = np.zeros((n, n), dtype=float)
            f = np.zeros((len(G.nodes()), n), dtype=float)
            for i,u in enumerate(G.nodes()):

                # 将节点链接状态转换为ndarray
                # n = len(util.node_dict(G)[u])
                arrays = np.zeros(n, dtype=float)

                # 填充连接情况到数组列表
                for ip, link in sorted(util.node_dict(G)[u].items()):
                    if link == 1:
                        arrays[ip] = 1



                # f[i,:] = util.node_dict(G)[u]
                f[i,:] = arrays
            self.feature_all.append(f)
            # else:
            # self.feature_all = self.deg_feat_all

        if features == 'default':
            self.feat_dim = self.feature_all[0].shape[1]
        self.node_dim = self.deg_feat_all[0].shape[0]
        self.deg_feat_dim = self.deg_feat_all[0].shape[1]

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        if self.max_num_nodes > num_nodes:
            adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
            adj_padded[:num_nodes, :num_nodes] = adj
        elif self.max_num_nodes < num_nodes:
            degs = np.sum(np.array(adj), 1)
            deg_index = np.argsort(degs, axis=0)
            deg_ind = deg_index[0:num_nodes-self.max_num_nodes]
            adj_padded = np.delete(adj, [deg_ind], axis=0)
            adj_padded = np.delete(adj_padded, [deg_ind], axis=1)
        else:
            adj_padded = adj
               
        return {'adj':adj_padded,
                'feats':self.feature_all[idx].copy(),
                'label':self.label_all[idx],
                'num_nodes': num_nodes,
                'deg_feats':self.deg_feat_all[idx].copy()}