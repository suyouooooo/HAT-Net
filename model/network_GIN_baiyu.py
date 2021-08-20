import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, DenseGINConv, dense_mincut_pool, global_add_pool
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
        #if not self.channel_last:
        #    print(x.shape, 111)
        #    x = x.permute(0, 2, 1)
        if x.dim() == 3:
            x = x.permute(0, 2, 1)

        x = self.bn(x)

        if x.dim() == 3:
            x = x.permute(0, 2, 1)

        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        #if not self.channel_last:
        #    x = x.permute(0, 2, 1)

        return  x

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

        self.lin1 = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU())
        self.dropout = nn.Dropout()
        self.lin2 = nn.Linear(dim, out_channels)

    def forward(self, x, edge_index, batch):
        outputs = []
        x = self.conv1(x, edge_index)
        outputs.append(x)
        x = self.conv2(x, edge_index)
        outputs.append(x)

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)
        s = self.pool1(x.reshape(-1, 64))
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)
        outputs.append(x)

        #print(x.shape, 'x.shape')
        #print(adj.shape, adj.dtype)
        #x = x.permute(0, 2, 1)
        x = self.conv3(x, adj)
        x = self.conv4(x, adj)
        #x = x.permute(0, 2, 1)
        #print(x.shape)

        s = self.pool2(x)
        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)


        x = self.conv5(x, adj)
        x = x.mean(dim=1)
        x = self.lin1(x)
        x = self.dropout(x)
        x = self.lin2(x)

        #print(x, mc1 + mc2, o1 + o2)
        #sys.exit()
        return x, mc1 + mc2, o1 + o2


        #x = self.conv1(x, edge_index)
        #x = self.conv2(x, edge_index)
        #x = self.conv3(x, edge_index)
        #x = self.conv4(x, edge_index)
        #x = self.conv5(x, edge_index)
        #x = global_add_pool(x, batch)
        #x = self.lin1(x).relu()
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.lin2(x)

        #return x
        #if self.training:
        #    cls_loss = F.cross_entropy(output, label, size_average=True)
        #    return output, cls_loss
        #return output

        #return F.log_softmax(x, dim=-1)
