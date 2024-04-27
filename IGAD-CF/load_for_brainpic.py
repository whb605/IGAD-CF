import networkx as nx
import numpy as np

# 将属性特征矩阵转化为图列表
def matrix_to_graph_list(adj_matrix, node_features):
    graph_list = []
    for i in range(adj_matrix.shape[0]):
        G = nx.Graph()
        # 添加节点
        for j in range(adj_matrix.shape[1]):
            G.add_node(j)
        # 添加节点属性特征
        for j in range(adj_matrix.shape[1]):
            G.nodes[j]['feat'] = node_features[i, j]
        # 添加边
        for j in range(adj_matrix.shape[1]):
            for k in range(j+1, adj_matrix.shape[1]):
                if adj_matrix[i,j,k] != 0:
                    G.add_edge(j, k)
        graph_list.append(G)
    return graph_list



