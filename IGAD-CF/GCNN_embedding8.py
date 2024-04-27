import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import util


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y




class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, deg_dim, node_dim, num_layers,
                 concat=False, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.att_dim = input_dim

        self.bias = True
        if args is not None:
            self.bias = args.bias

        if input_dim > 0:
            self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim // 2, embedding_dim // 2, num_layers,
                add_self, normalize=True, dropout=dropout)
            self.conv_deg_first, self.conv_deg_block, self.conv_deg_last = self.build_conv_layers(
                deg_dim, hidden_dim // 2, embedding_dim // 2, num_layers,
                add_self, normalize=True, dropout=dropout)
        else:
            self.conv_deg_first, self.conv_deg_block, self.conv_deg_last = self.build_conv_layers(
                deg_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)

        self.act = nn.ReLU()
        self.Embed = nn.Linear(embedding_dim * node_dim, embedding_dim)
        self.Encode = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Decode = nn.Linear(embedding_dim, 1)
        self.acts = nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
                          normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                               normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                       normalize_embedding=normalize, dropout=dropout, bias=self.bias)
             for i in range(num_layers - 2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                              normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self, x0, x1, adj):
        if self.att_dim > 0:
            x = self.conv_first(x0, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            for i in range(self.num_layers - 2):
                x = self.conv_block[i](x, adj)
                x = self.act(x)
                if self.bn:
                    x = self.apply_bn(x)
            x0 = self.conv_last(x, adj)
        x = self.conv_deg_first(x1, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        for i in range(self.num_layers - 2):
            x = self.conv_deg_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
        x1 = self.conv_deg_last(x, adj)
        if self.att_dim > 0:
            x = torch.cat([x0, x1], dim=-1)
        else:
            x = x1
        x = util.sort_tensor_by_distance(x)
        x = self.Embed(torch.reshape(x, (len(x), -1)))
        w = self.Encode(x)
        x = x * w
        x = self.act(x)
        x = self.Decode(x)
        out = self.acts(x)

        return out

