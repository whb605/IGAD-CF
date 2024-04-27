"""
@author: LinFu
"""
import os

from torch_geometric.data import Data
from GCG import generation
import transverter
import scipy.io
import numpy as np
from random import sample
from sklearn.metrics import auc, precision_recall_curve, roc_curve
import argparse

import torch
import torch.nn as nn
import GCNN_embedding8
from torch.autograd import Variable


from graph_sampler_b import GraphSampler
from numpy.random import seed
import random

from sklearn.model_selection import StratifiedKFold
import warnings

import pickle



warnings.filterwarnings("ignore")


def save_generated_data(dataname, fn, data):
    path = f'./data/' + dataname + fn
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_generated_data(dataname, fn):
    path = f'./data/' + dataname + fn
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def output_num(num):
    return round(num * 100, 2)


def edge_index_to_adj(edge_index, num_nodes):
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    return adj


def arg_parse():
    parser = argparse.ArgumentParser(description='GLADD Arguments.')
    parser.add_argument('--datadir', dest='datadir', default="./brain_data/", help='Directory where benchmark is located')
    parser.add_argument('--DS', dest='DS', default="BP.mat", help='dataset name')
    parser.add_argument('--X_ray_method', dest='X_ray_method', default="dti", help='X_ray_method')
    parser.add_argument('--brain_threshold', dest='threshold', default=0.5, help='threshold in brain graph')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int, default=0,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--clip', dest='clip', default=0.1, type=float, help='Gradient clipping.')
    parser.add_argument('--lr', dest='lr', default=0.00001, type=float, help='Learning Rate.')
    parser.add_argument('--num_epochs', dest='num_epochs', default=100, type=int, help='total epoch number')
    parser.add_argument('--batch-size', dest='batch_size', default=800, type=int, help='Batch size.')
    parser.add_argument('--hidden-dim', dest='hidden_dim', default=512, type=int, help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', default=256, type=int, help='Output dimension')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', default=3, type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const', const=True, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', default=0.5, type=float, help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const', const=True, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='seed')
    parser.add_argument('--sign', dest='sign', type=int, default=1, help='sign of graph anomaly')
    parser.add_argument('--feature', dest='feature', default='default', help='use what node feature')
    parser.add_argument('--cnn', dest='cnn', default=False, help='Whether CNN is used')
    parser.add_argument('--beta', dest='beta', default=0.6, help='beta')
    parser.add_argument('--node_features', dest='node_features', default="identity", help='node_features')
    return parser.parse_args()

def arg_parse_CGC():
    parser_GCG = argparse.ArgumentParser()
    parser_GCG.add_argument('--cuda', type=bool, default=True)
    parser_GCG.add_argument('--random_seed', type=int, default=12345)
    parser_GCG.add_argument('--gnn_layers_num', type=int, default=3)
    parser_GCG.add_argument('--gnn', type=str, default='GIN')
    parser_GCG.add_argument('--generation_lr', type=float, default=1e-3)
    parser_GCG.add_argument('--pre_lr', type=float, default=1e-3)
    parser_GCG.add_argument('--generation_epochs', type=int, default=30)
    parser_GCG.add_argument('--pre_epochs', type=int, default=30)
    parser_GCG.add_argument('--batch_size', type=int, default=256)
    parser_GCG.add_argument('--temp', type=float, default=1)
    parser_GCG.add_argument('--gamma', type=float, default=0.3)
    return parser_GCG.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compare_dicts(dict1, dict2):
    for key in dict1:
        if key not in dict2 or not torch.equal(dict1[key], dict2[key]):
            return False

    return True


def train(dataset_train_p, dataset_train_n, dataset_test, model, args):

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    max_AUC = 0

    auroc_final = 0

    best_fpr = None
    best_tpr = None
    data_p = []
    data_n = []
    for batch_idx, data in enumerate(dataset_train_p):
        data_p.append(data)
    for batch_idx, data in enumerate(dataset_train_n):
        data_n.append(data)
    datasize_p = len(data_p)
    datasize_n = len(data_n)


    for epoch in range(args.num_epochs):
        model.train()
        current_loss = 0
        for idx in range(datasize_p):
            model.zero_grad()
            adj_p = Variable(data_p[idx]['adj'].float(), requires_grad=False).cuda()
            h0_p = Variable(data_p[idx]['feats'].float(), requires_grad=False).cuda()
            h1_p = Variable(data_p[idx]['deg_feats'].float(), requires_grad=False).cuda()
            adj_n = Variable(data_n[idx]['adj'].float(), requires_grad=False).cuda()
            h0_n = Variable(data_n[idx]['feats'].float(), requires_grad=False).cuda()
            h1_n = Variable(data_n[idx]['deg_feats'].float(), requires_grad=False).cuda()
            loss_ = model(torch.cat([h0_p, h0_n]), torch.cat([h1_p, h1_n]), torch.cat([adj_p, adj_n]))
            loss1 = loss_[:len(adj_p)]
            loss2 = loss_[len(adj_p):]
            if alpha > 0:
                loss_R = loss1[:int((1 - alpha) * len(adj_p))]
                loss_HN = loss1[int((1 - alpha) * len(adj_p)):]
                loss_HN = -torch.log(1 - loss_HN).mean()
                loss1 = -torch.log(1 - loss_R).mean()
                loss2 = -torch.log(loss2).mean()

                loss = (loss1 * ((1 - alpha)) + alpha * loss_HN * args.beta) * (0.5) + loss2 * (0.5)


            elif alpha < 0:

                loss_R = loss2[:int((1 + alpha) * len(adj_n))]
                loss_HN = loss2[int((1 + alpha) * len(adj_n)):]
                loss_HN = -torch.log(loss_HN).mean()
                loss2 = -torch.log(loss_R).mean()
                loss1 = -torch.log(1 - loss1).mean()
                loss = loss1 * (0.5) + (loss2 * (1 + alpha) + loss_HN *(-alpha)* args.beta)*0.5

            else:
                loss1 = -torch.log(1 - loss1).mean()
                loss2 = -torch.log(loss2).mean()
                loss = loss1 * (0.5) + loss2 * (0.5)

            current_loss = loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

        if (epoch + 1) % 10 == 0 and epoch > 0:
            model.eval()
            loss = []
            y = []

            for batch_idx, data in enumerate(dataset_test):
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
                h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
                h1 = Variable(data['deg_feats'].float(), requires_grad=False).cuda()
                loss_ = model(h0, h1, adj)
                loss_ = np.array(loss_.cpu().detach())
                loss.extend(loss_)

                for data_label in data['label']:
                    if data_label == args.sign:
                        y.append(1)
                    else:
                        y.append(0)

            label_test = np.array(loss).squeeze()
            try:

                fpr_ab, tpr_ab, thr_ = roc_curve(y, label_test)


                test_roc_ab = auc(fpr_ab, tpr_ab)

                print('abnormal detection: auroc_ab: {}, loss: {}'.format(test_roc_ab, current_loss))
                if test_roc_ab > max_AUC:
                    max_AUC = test_roc_ab

            except Exception as e:print(e)

        if epoch == (args.num_epochs - 1):
            auroc_final = max_AUC



    return auroc_final


if __name__ == '__main__':
    args = arg_parse()
    args_GCG = arg_parse_CGC()
    device = 'cuda'
    setup_seed(args.seed)
    DSS=args.DS
    X_ray_method=args.X_ray_method
    threshold=args.threshold

    mat_path = args.datadir + DSS
    mat_contents = scipy.io.loadmat(mat_path)
    fmri_data = mat_contents[X_ray_method]
    graphs_label1 = mat_contents['label']
    graphs_label1 = graphs_label1.tolist()
    node_labels = []
    label_0_num = 0
    for i in graphs_label1:
        if i[0] == 1:
            node_labels.append(i[0])
        else:
            node_labels.append(0)
            label_0_num += 1

    adj_matrix = np.where(fmri_data > threshold, 1, 0)
    adj_matrix = adj_matrix.transpose((2, 1, 0))
    feature_computed = transverter.compute_x(torch.from_numpy(adj_matrix), args.node_features)

    pic_num = adj_matrix.shape[0]
    node_feats_list = []

    node_feats = feature_computed

    graphs = transverter.convert_to_networkx2(adj_matrix, node_feats, node_labels)
    graphs_label = node_labels
    datanum = len(graphs)
    if args.max_nodes == 0:
        max_nodes_num = max([G.number_of_nodes() for G in graphs])
    else:
        max_nodes_num = args.max_nodes


    print('GraphNumber: {}'.format(datanum))


    max_nodes = max_nodes_num

    feature_computed = transverter.compute_x(torch.from_numpy(adj_matrix), args.node_features)
    pic_num = adj_matrix.shape[0]
    node_feats_list = []
    node_feats = feature_computed
    loaded_datas = []

    node_feats = node_feats.to(device)

    for i in range(pic_num):
        edge_index = torch.tensor(np.nonzero(adj_matrix[i])).t().contiguous().long().t()
        edge_index = edge_index.to(device)
        data = Data(
            x=torch.tensor(node_feats[i]).to(device),
            edge_index=torch.tensor(edge_index).to(device),
            y=torch.tensor(graphs_label[i]).to(device),
            id=i,
        )
        loaded_datas.append(data)
    datanum = len(loaded_datas)

    max_nodes_num = max_nodes


    kfd = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)
    result_auc = []
    exp_accs = []
    exp_macros = []


    graphs_num = len(loaded_datas)
    attrs_dim = loaded_datas[0].x.size()[1]
    nodes_num_list = [max_nodes_num] * graphs_num
    if os.path.exists(f"./data/" + DSS+str(threshold) + args.node_features + 'trainCGC_list.pkl'):
        print("loading...")
        CGC_list = load_generated_data(DSS +str(threshold)+ args.node_features, 'trainCGC_list.pkl')

    else:
        CGC_list = generation(args_GCG, graphs_num, nodes_num_list, attrs_dim, loaded_datas)
        save_generated_data(DSS+str(threshold) + args.node_features, 'trainCGC_list.pkl', CGC_list)

    for k, (train_index, test_index) in enumerate(kfd.split(loaded_datas, graphs_label)):


        graphs_train = [loaded_datas[i] for i in train_index]
        CGC_list_train = [CGC_list[i] for i in train_index]
        graphs_test = [loaded_datas[i] for i in test_index]

        labels_train = [graphs_label[i] for i in train_index]
        labels_test = [graphs_label[i] for i in test_index]

        graphs_train_n = []
        graphs_train_p = []

        CGC_list_n = []
        CGC_list_p = []


        labels_train_n = []
        labels_train_p = []

        for igraph, graph in enumerate(graphs_train):
            if labels_train[igraph] != args.sign:
                graphs_train_p.append(graph)
                CGC_list_n.append(CGC_list_train[igraph])
                labels_train_p.append(labels_train[igraph])

            else:
                graphs_train_n.append(graph)
                CGC_list_p.append(CGC_list_train[igraph])
                labels_train_n.append(labels_train[igraph])

        graphs_test_p = []
        graphs_test_n = []
        for igraph, graph in enumerate(graphs_test):
            if labels_train[igraph] != args.sign:
                graphs_test_p.append(graph)
            else:
                graphs_test_n.append(graph)

        datasize_p = len(graphs_train_p)
        datasize_n = len(graphs_train_n)

        alpha = int(datasize_n - datasize_p) / max(datasize_n, datasize_p)




        if datasize_p < datasize_n:
            sample_num = int(datasize_n - datasize_p)
            labels_train_p.extend([labels_train_p[0]] * sample_num)
            graphs_train_p.extend(sample(CGC_list_p, sample_num))
        if datasize_p > datasize_n:
            sample_num = int(datasize_p - datasize_n)
            labels_train_n.extend([labels_train_n[0]] * sample_num)
            graphs_train_n.extend(sample(CGC_list_n, sample_num))

        adj_matrixs = []
        node_fs = []
        for ii, ig in enumerate(graphs_train_p):
            adj_matrix = edge_index_to_adj(ig.edge_index, max_nodes_num)
            adj_matrixs.append(adj_matrix)
            node_fs.append(ig.x)

        adj_matrixs = [tensor.numpy() for tensor in adj_matrixs]
        adj_matrixs = np.array(adj_matrixs)
        device = 'cuda'
        node_fs = [tensor.to(device) for tensor in node_fs]

        node_fs = torch.stack(node_fs, dim=0)

        graphs_train_p = transverter.convert_to_networkx2(adj_matrixs, node_fs, labels_train_p)

        adj_matrixs = []
        node_fs = []
        for ii, ig in enumerate(graphs_train_n):
            adj_matrix = edge_index_to_adj(ig.edge_index, max_nodes_num)
            adj_matrixs.append(adj_matrix)
            node_fs.append(ig.x)

        adj_matrixs = [tensor.numpy() for tensor in adj_matrixs]
        adj_matrixs = np.array(adj_matrixs)
        device = 'cuda'
        node_fs = [tensor.to(device) for tensor in node_fs]

        node_fs = torch.stack(node_fs, dim=0)
        graphs_train_n = transverter.convert_to_networkx2(adj_matrixs, node_fs, labels_train_n)

        adj_matrixs = []
        node_fs = []
        for ii, ig in enumerate(graphs_test):
            adj_matrix = edge_index_to_adj(ig.edge_index, max_nodes_num)
            adj_matrixs.append(adj_matrix)
            node_fs.append(ig.x)

        adj_matrixs = [tensor.numpy() for tensor in adj_matrixs]
        adj_matrixs = np.array(adj_matrixs)
        device = 'cuda'
        node_fs = [tensor.to(device) for tensor in node_fs]

        node_fs = torch.stack(node_fs, dim=0)
        graphs_test = transverter.convert_to_networkx2(adj_matrixs, node_fs, labels_test)

        num_train_p = len(graphs_train_p)
        num_train_n = len(graphs_train_n)
        num_test = len(graphs_test)
        print('TrainSize_p: {}, TrainSize_n: {}, TestSize: {}'.format(num_train_p, num_train_n,
                                                                      num_test))
        dataset_sampler_train_p = GraphSampler(graphs_train_p, labels=labels_train_p,
                                               features=args.feature,
                                               normalize=False,
                                               max_num_nodes=max_nodes_num)
        data_train_loader_p = torch.utils.data.DataLoader(dataset_sampler_train_p, shuffle=True,
                                                          batch_size=args.batch_size)
        dataset_sampler_train_n = GraphSampler(graphs_train_n, labels=labels_train_n,
                                               features=args.feature,
                                               normalize=False,
                                               max_num_nodes=max_nodes_num)
        data_train_loader_n = torch.utils.data.DataLoader(dataset_sampler_train_n, shuffle=True,
                                                          batch_size=args.batch_size)
        dataset_sampler_test = GraphSampler(graphs_test, labels=labels_test, features=args.feature,
                                            normalize=False,
                                            max_num_nodes=max_nodes_num)
        data_test_loader = torch.utils.data.DataLoader(dataset_sampler_test, shuffle=False,
                                                       batch_size=args.batch_size)

        model = GCNN_embedding8.GcnEncoderGraph(dataset_sampler_train_p.feat_dim,
                                                args.hidden_dim,
                                                args.output_dim,
                                                dataset_sampler_train_p.deg_feat_dim,
                                                dataset_sampler_train_p.node_dim,
                                                args.num_gc_layers,
                                                bn=args.bn,
                                                dropout=args.dropout, args=args).cuda()

        result= train(data_train_loader_p, data_train_loader_n, data_test_loader,
                                     model,
                                     args)

        result_auc.append(result)


    result_auc = np.array(result_auc)
    auc_avg = np.mean(result_auc)
    auc_std = np.std(result_auc)

    print("---Datasets:"+args.DS+"---Average AUC:"+str(auc_avg)+"---Standard deviation:"+str(auc_std))

