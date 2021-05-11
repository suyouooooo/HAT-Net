import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv,DenseGINConv, DenseGCNConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch
# import torch.utils.checkpoint as cp
# from torch_geometric.utils import to_dense_batch
# from model.utils import to_dense_adj
from model.utils import to_dense_batch
from torch_geometric.utils import to_dense_adj
from torch.nn import Linear, LSTM
EPS = 1e-15
import pdb


class DenseJK(nn.Module):
    def __init__(self, mode, channels=None, num_layers=None):
        super(DenseJK, self).__init__()
        self.channel = channels
        self.mode = mode.lower()
        assert self.mode in ['cat', 'max', 'lstm']

        if mode == 'lstm':
            assert channels is not None
            assert num_layers is not None
            self.lstm = LSTM(
                channels,
                channels  * num_layers // 2,
                bidirectional=True,
                batch_first=True)
            self.att = Linear(2 * channels * num_layers // 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lstm'):
            self.lstm.reset_parameters()
        if hasattr(self, 'att'):
            self.att.reset_parameters()

    def forward(self, xs):
        xs = torch.split(xs, self.channel, -1)# list of batch, node, featdim

        if self.mode == 'lstm':
            xs = torch.stack(xs, 2)  # [batch, nodes, num_layers, num_channels]
            shape = xs.shape
            x = xs.reshape((-1, shape[2], shape[3]))  # [ngraph * num_nodes , num_layers, num_channels]
            self.lstm.flatten_parameters()
            alpha, _ = self.lstm(x)
            alpha = self.att(alpha).squeeze(-1)  # [ngraph * num_nodes, num_layers]
            alpha = torch.softmax(alpha, dim=-1)
            x =  (x * alpha.unsqueeze(-1)).sum(dim=1)
            x = x.reshape((shape[0],shape[1],shape[3]))
            return x

        elif self.mode == 'cat':
            return torch.cat(xs, dim=-1)

        elif self.mode == 'max':
            return torch.stack(xs, dim=-1).max(dim=-1)[0]



    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.mode)


class SoftPoolingGcnEncoder(nn.Module):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, bias, bn, assign_hidden_dim,label_dim,
                 assign_ratio=0.25,  pred_hidden_dims=[50], concat = True, gcn_name='SAGE',
                 collect_assign = False, load_data_sparse = False,norm_adj=False,
                 activation = 'relu',  drop_out = 0.,jk = False):


        super(SoftPoolingGcnEncoder, self).__init__()

        self.jk = jk
        self.drop_out = drop_out
        self.norm_adj = norm_adj
        self.load_data_sparse = load_data_sparse
        self.collect_assign = collect_assign
        self.assign_matrix = []
        assign_dim = int(max_num_nodes * assign_ratio)
        # self.GCN_embed_1 = GNN_Module(input_dim, hidden_dim, embedding_dim, bias, bn,
        #                               add_loop= False, lin=False, gcn_name=gcn_name,activation=activation, jk = jk)
        nn1 = nn.Sequential(nn.Linear(input_dim, embedding_dim), self._activation(activation),
                            nn.Linear(embedding_dim, embedding_dim))
        self.GCN_embed_1 = DenseGINConv(nn1)
        if jk:
            self.jk1 = DenseJK('lstm', hidden_dim, 3)

        # self.GCN_pool_1 = GNN_Module(input_dim, assign_hidden_dim, assign_dim, bias, bn,
        #                              add_loop= False, gcn_name=gcn_name,activation=activation, jk = jk)

        nn2 = nn.Sequential(nn.Linear(input_dim, assign_dim), self._activation(activation),
                            nn.Linear(assign_dim, assign_dim))
        self.GCN_pool_1 = DenseGINConv(nn2)

        if concat and not jk:
            input_dim = hidden_dim * 2 + embedding_dim
        else:
            input_dim = embedding_dim

        assign_dim = int(assign_dim * assign_ratio)
        # self.GCN_embed_2 = GNN_Module(input_dim, hidden_dim, embedding_dim, bias, bn,
        #                               add_loop= False,lin=False, gcn_name=gcn_name,activation=activation, jk = jk)
        nn3 = nn.Sequential(nn.Linear(input_dim, embedding_dim), self._activation(activation),
                            nn.Linear(embedding_dim, embedding_dim))
        self.GCN_embed_2 = DenseGINConv(nn3)
        if jk:
            self.jk2 = DenseJK('lstm', hidden_dim, 3)
        # self.GCN_pool_2 = GNN_Module(input_dim, assign_hidden_dim, assign_dim, bias, bn,
        #                              add_loop= False, gcn_name=gcn_name,activation=activation, jk = jk)

        nn4 = nn.Sequential(nn.Linear(input_dim, assign_dim), self._activation(activation),
                            nn.Linear(assign_dim, assign_dim))
        self.GCN_pool_2 = DenseGINConv(nn4)

        # self.GCN_embed_3 = GNN_Module(input_dim, hidden_dim, embedding_dim, bias, bn,
        #                               add_loop= False,lin=False, gcn_name=gcn_name,activation=activation, jk = jk)

        nn5 = nn.Sequential(nn.Linear(input_dim, embedding_dim), self._activation(activation),
                            nn.Linear(embedding_dim, embedding_dim))
        self.GCN_embed_3 = DenseGINConv(nn5)
        if jk:
            self.jk3 = DenseJK('lstm', hidden_dim, 3)

        nn6 = nn.Sequential(nn.Linear(input_dim, assign_dim), self._activation(activation),
                            nn.Linear(assign_dim, assign_dim))
        self.GCN_pool_3 = DenseGINConv(nn6)


        pred_input = input_dim * 3
        self.pred_model = self.build_readout_module(pred_input, pred_hidden_dims,
                                                    label_dim, activation)

        self.mlp = nn.Sequential(
            nn.Linear(pred_input , pred_input), # pred_input 60
            nn.ReLU(),
            nn.Linear(pred_input, pred_input // 2),
            nn.ReLU(),
            nn.Linear(pred_input // 2, label_dim))


    @staticmethod
    def construct_mask( max_nodes, batch_num_nodes):
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()


    def _re_norm_adj(self,adj,p, mask = None):
        # pdb.set_trace()
        idx = torch.arange(0, adj.shape[1],out=torch.LongTensor())
        adj[:,idx,idx] = 0
        new_adj =  torch.div(adj,adj.sum(-1)[...,None] + EPS)*(1-p)
        new_adj[:,idx,idx] = p
        if mask is not None:
            new_adj = new_adj * mask
        return new_adj


    def _diff_pool(self, x, adj, s, mask):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s
        batch_size, num_nodes, _ = x.size()
        s = torch.softmax(s, dim=-1)
        if self.collect_assign:
            self.assign_matrix.append(s.detach())
        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            s = s * mask
        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        return  out, out_adj


    def _rank3_trace(self, x):
        return torch.einsum('ijj->i', x)

    def _rank3_diag(self, x):
        eye = torch.eye(x.size(1)).type_as(x)
        out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
        return out

    def dense_mincut_pool(self, x, adj, s, mask):

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        (batch_size, num_nodes, _), k = x.size(), s.size(-1)

        s = torch.softmax(s, dim=-1)

        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            x, s = x * mask, s * mask

        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        # MinCUT regularization.
        mincut_num = self._rank3_trace(out_adj)
        d_flat = torch.einsum('ijk->ij', adj)
        d = self._rank3_diag(d_flat)
        mincut_den = self._rank3_trace(
            torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
        mincut_loss = -(mincut_num / mincut_den)
        mincut_loss = torch.mean(mincut_loss)

        # Orthogonality regularization.
        ss = torch.matmul(s.transpose(1, 2), s)
        i_s = torch.eye(k).type_as(ss)
        ortho_loss = torch.norm(
            ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
            i_s / torch.norm(i_s), dim=(-1, -2))
        ortho_loss = torch.mean(ortho_loss)

        # Fix and normalize coarsened adjacency matrix.
        ind = torch.arange(k, device=out_adj.device)
        out_adj[:, ind, ind] = 0
        d = torch.einsum('ijk->ij', out_adj)
        d = torch.sqrt(d)[:, None] + EPS
        out_adj = (out_adj / d) / d.transpose(1, 2)

        return out, out_adj, mincut_loss, ortho_loss, num_nodes


    def _activation(self, name = 'relu'):
        assert name in ['relu', 'elu', 'leakyrelu']
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'elu':
            return nn.ELU(inplace=True)
        elif name =='leakyrelu':
            return nn.LeakyReLU(inplace=True)

    def build_readout_module(self,pred_input_dim, pred_hidden_dims, label_dim, activation ):
        pred_input_dim = pred_input_dim
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self._activation(activation))
                pred_input_dim = pred_dim
                if self.drop_out>0:
                    pred_layers.append(nn.Dropout(self.drop_out))
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model


    def _sparse_to_dense_input(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        label = data.y
        edge_index = to_dense_adj(edge_index, batch)
        x ,batch_num_node= to_dense_batch(x, batch)
        return x, edge_index,batch_num_node,label

    def forward(self,  data):
        out_all = []
        self.assign_matrix = []
        if self.load_data_sparse:
            x, adj, batch_num_nodes, label = self._sparse_to_dense_input(data)
        else:
            x, adj, batch_num_nodes= data[0], data[1], data[2]
            if self.training:
                label = data[3]
        max_num_nodes = adj.size()[1]
        embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        # print("stage 0: " + str(x.size()))
        # print("stage 0 adj: " + str(adj.size()))


        # stage 1
        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4, embedding_mask)
        embed_feature = self.GCN_embed_1(x, adj, embedding_mask)
        if self.jk:
            embed_feature = self.jk1(embed_feature)
        # out, _ = torch.max(embed_feature, dim = 1)
        # out_all.append(out)
        assign = self.GCN_pool_1(x, adj, embedding_mask)
        x, adj,mincut_loss, ortho_loss, num_nodes = self.dense_mincut_pool(embed_feature, adj, assign, embedding_mask)
        # print("stage 1: " + str(x.size()))
        # print("stage 1 adj: " + str(adj.size()))
        out, _ = torch.max(x, dim=1)
        out_all.append(out)

        # # new readout
        # # change int num_node to Long Tensor
        # num_nodes_list = []
        # num_nodes_list.append(num_nodes)
        # num_nodes = torch.LongTensor(num_nodes_list).cuda()
        # x1 = torch.cat([gmp(x, batch_num_nodes), gap(x, batch_num_nodes)], dim=1) # x1 size torch.Size([2481, 2280, 20])


        # stage 2
        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4)
        embed_feature = self.GCN_embed_2(x, adj, None)
        if self.jk:
            embed_feature = self.jk2(embed_feature)
        # out, _ = torch.max(embed_feature, dim=1)
        # out_all.append(out)
        assign = self.GCN_pool_2(x, adj, None)

        x, adj,mincut_loss, ortho_loss, num_nodes = self.dense_mincut_pool(embed_feature, adj, assign, None)
        # print("stage 2: " + str(x.size()))
        # print("stage 2 adj: " + str(adj.size()))
        out, _ = torch.max(x, dim=1)
        out_all.append(out)

        # # new readout
        # # change int num_node to Long Tensor
        # num_nodes_list = []
        # num_nodes_list.append(num_nodes)
        # num_nodes = torch.LongTensor(num_nodes_list).cuda()
        # x2 = torch.cat([gmp(x, batch_num_nodes), gap(x, batch_num_nodes)], dim=1) # x2 size torch.Size([1141, 228, 20])


        # stage 3
        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4)
        embed_feature = self.GCN_embed_3(x, adj, None)

        if self.jk:
            embed_feature = self.jk3(embed_feature)

        # out, _ = torch.max(embed_feature, dim=1)
        # out_all.append(out)
        assign = self.GCN_pool_3(x, adj, None)
        x, adj, mincut_loss, ortho_loss, num_nodes = self.dense_mincut_pool(embed_feature, adj, assign, None)
        # print("stage 3: " + str(x.size()))
        # print("stage 3 adj: " + str(adj.size()))
        out, _ = torch.max(x, dim=1)
        out_all.append(out)  # out_all[0].size() torch.Size([1, 20])
        output = torch.cat(out_all, 1) # output.size() torch.Size([1, 60])

        # output = self.pred_model(output)

        output = self.mlp(output)

        # # new readout
        # # change int num_node to Long Tensors
        # num_nodes_list = []
        # num_nodes_list.append(num_nodes)
        # num_nodes = torch.LongTensor(num_nodes_list).cuda()
        # x3 = torch.cat([gmp(x, batch_num_nodes), gap(x, batch_num_nodes)], dim=1) # x3 size torch.Size([115, 228, 20])
        #
        # readout = x1 + x2 + x3
        # output = self.mlp(readout)


        if self.training:
            cls_loss = F.cross_entropy(output, label, size_average=True)
            return output, cls_loss
        return output
