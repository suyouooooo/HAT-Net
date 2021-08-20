import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, DenseGINConv, dense_mincut_pool, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj


class NN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()
        # input channel tensor

    def forward(self, x):
        x = self.fc1(x)
        if x.dim() == 3:
            x = x.permute(0, 2, 1)

        x = self.bn(x)

        if x.dim() == 3:
            x = x.permute(0, 2, 1)

        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        return  x

#class HatNet(torch.nn.Module):
#    def __init__(self, in_channels, dim, out_channels):
#        #super(Net, self).__init__()
#        super().__init__()
#        from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
#
#        self.conv1 = GINConv(
#            Sequential(Linear(in_channels, dim), BatchNorm1d(dim), ReLU(),
#                       Linear(dim, dim), ReLU()))
#
#        self.conv2 = GINConv(
#            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
#                       Linear(dim, dim), ReLU()))
#
#        self.conv3 = GINConv(
#            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
#                       Linear(dim, dim), ReLU()))
#
#        self.conv4 = GINConv(
#            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
#                       Linear(dim, dim), ReLU()))
#
#        self.conv5 = GINConv(
#            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
#                       Linear(dim, dim), ReLU()))
#
#        self.lin1 = Linear(dim, dim)
#        self.lin2 = Linear(dim, out_channels)
#
#    def forward(self, x, edge_index, batch):
#        outputs = []
#        x = self.conv1(x, edge_index)
#
#        #for i in outputs:
#            #print(i.shape)
#
#        #print()
#        x = self.conv2(x, edge_index)
#        x = self.conv3(x, edge_index)
#        x = self.conv4(x, edge_index)
#        x = self.conv5(x, edge_index)
#        print(x.shape)
#        x = global_add_pool(x, batch)
#        print(x.shape)
#        x = self.lin1(x).relu()
#        x = F.dropout(x, p=0.5, training=self.training)
#        x = self.lin2(x)
#        print(x.shape)
#        import sys; sys.exit()
#        return F.log_softmax(x, dim=-1)

class HatNet(nn.Module):
    def __init__(self, in_channels, dim, out_channels, num_nodes=548):
        super().__init__()
        #num_nodes = num_nodes
        num_nodes = int(0.5 * num_nodes)
        self.pool1 = nn.Linear(dim, num_nodes)
        num_nodes = int(0.5 * num_nodes)
        self.pool2 = nn.Linear(dim, num_nodes)

        #self.conv1 = GINConv(
        #    nn.Sequential(nn.Linear(in_channels, dim), nn.BatchNorm1d(dim), nn.ReLU(),
        #               nn.Linear(dim, dim), nn.ReLU()))
        self.conv1 = GINConv(NN(in_channels, dim))
        self.conv2 = GINConv(NN(dim, dim))
        self.conv3 = DenseGINConv(NN(dim, dim))
        self.conv4 = DenseGINConv(NN(dim, dim))
        self.conv5 = DenseGINConv(NN(dim, dim))

        #self.conv2 = GINConv(
        #    nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
        #               nn.Linear(dim, dim), nn.ReLU()))

        #self.conv3 = DenseGINConv(
        #    nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
        #               nn.Linear(dim, dim), nn.ReLU()))

        #self.conv4 = DenseGINConv(
        #    nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
        #               nn.Linear(dim, dim), nn.ReLU()))

        #self.conv5 = DenseGINConv(
        #    nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
        #               nn.Linear(dim, dim), nn.ReLU()))

        self.fc1 = nn.Sequential(nn.Linear(dim * 21, dim * 21 // 4), nn.BatchNorm1d(dim * 21 // 4), nn.ReLU())
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(dim * 21 // 4, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        outputs = []
        x = self.conv1(x, edge_index)
        outputs.append(global_add_pool(x, batch))
        outputs.append(global_mean_pool(x, batch))
        outputs.append(global_max_pool(x, batch))

        x = self.conv2(x, edge_index)
        outputs.append(global_add_pool(x, batch))
        outputs.append(global_mean_pool(x, batch))
        outputs.append(global_max_pool(x, batch))

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)
        s = self.pool1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)
        outputs.append(torch.mean(x, dim=1))
        outputs.append(torch.sum(x, dim=1))
        outputs.append(torch.max(x, dim=1)[0])

        x = self.conv3(x, adj)
        outputs.append(torch.mean(x, dim=1))
        outputs.append(torch.sum(x, dim=1))
        outputs.append(torch.max(x, dim=1)[0])

        x = self.conv4(x, adj)
        outputs.append(torch.mean(x, dim=1))
        outputs.append(torch.sum(x, dim=1))
        outputs.append(torch.max(x, dim=1)[0])

        s = self.pool2(x)
        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
        outputs.append(torch.mean(x, dim=1))
        outputs.append(torch.sum(x, dim=1))
        outputs.append(torch.max(x, dim=1)[0])

        x = self.conv5(x, adj)
        outputs.append(torch.mean(x, dim=1))
        outputs.append(torch.sum(x, dim=1))
        outputs.append(torch.max(x, dim=1)[0])

        x = torch.cat(outputs, dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        mc_loss = mc1 + mc2
        o_loss = o1 + o2
        loss = F.cross_entropy(x, data.y, size_average=True) + mc_loss + o_loss
        return x, loss
