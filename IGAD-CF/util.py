import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch

def node_iter(G):

    return G.nodes

def node_dict(G):

    return G.nodes


def sort_tensor_by_distance(tensor):

    distances = torch.norm(tensor, p=1, dim=2)

    sorted_indices = torch.argsort(-distances, dim=1)

    sorted_tensor = torch.gather(tensor, 1, sorted_indices.unsqueeze(2).expand(-1, -1, tensor.shape[2]))

    return sorted_tensor