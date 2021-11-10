import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, DenseGINConv, dense_mincut_pool, global_add_pool, global_mean_pool, global_max_pool, GCNConv, SAGPooling, SAGEConv, JumpingKnowledge
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
import torch_geometric
from torch_cluster import grid_cluster




class NN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()
        # input channel tensor

    def forward(self, x):
        #print(x.shape, self.fc1)
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

class SplitTrans(nn.Module):
    def __init__(self, encoder_layer, dim):
        super().__init__()
        self.trans_tl = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.trans_tr = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.trans_bl = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.trans_br = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pad_emb = nn.Embedding(1, dim, padding_idx=0)
        self.dim = dim

    #def
    #grid_cluster
    #def grid_cluster(self, batch):
    #    clusters = []
    #    #degs = []
    #    #print(batch.num_graphs)
    #    for i in range(batch.num_graphs):
    #    #for data in data_list:
    #        #print(data)
    #        data = batch[i]
    #        image_size = data.image_size
    #        h, w = image_size[:2]

    #        h = int(math.ceil(h / 2))
    #        w = int(math.ceil(w / 2))
    #        cluster = grid_cluster(data.pos, torch.tensor([h, w]).to(data.x.device))
    #        #print(cluster.max())
    #        #print(cluster.shape)
    #        #print(cluster.max())
    #        clusters.append(cluster)


    #    #return batch
    #    return clusters


    ##def degree(self, data_list):
    #def degree(self, batch):
    #    degs = []
    #    for i in range(batch.num_graphs):
    #    #for data in data_list:
    #        data = batch[i]
    #        #print(data)
    #        deg = degree(data.edge_index[1])
    #        degs.append(deg)
    #        #print(deg.max())

    #    return degs

    def batching(self, data_list):
        res = []
        num_nodes = [data.shape[0] for data in data_list]
        max_node = max(num_nodes)
        batch = torch.zeros((len(data_list), max_node, self.dim)).to(data_list[0].device)
        #mask = torch.zeros((len(data_list), max_node), dtype=torch.bool).to(data_list[0].device)
        #print(batch.shape)
        batch[:, :] = self.pad_emb(torch.tensor(0).to(batch.device))
        #print(batch)
        #print(batch.shape)
        #print(max_node)
        for idx, num_node in enumerate(num_nodes):
            batch[idx, :num_node] = data_list[idx]
            #mask[idx, :num_node] = True

        #print(mask.shape)
        #print(mask[4])
        #print(mask[3])
        #print(batch[0].shape)
        #print(batch.shape)
        return batch, num_nodes

    def assign_data(self, batch, node_feat, masks, num_nodes):
        #batch.x[tl_m] = tl[idx, :num_nodes]
        total = 0
        for idx, (m, n_node) in enumerate(zip(masks, num_nodes)):
            #print(m, n_node)
            node_size = m.size(0)
            #print(m.shape, 1111)
            #print(m.shape)
            #print(batch.x[total : total + node_size].shape, m.shape)
            #print(batch.x[total : total + node_size][m].shape, n_node)
            #print(node_feat[idx, :n_node].shape, node_feat.shape)
            batch.x[total : total + node_size][m] = node_feat[idx, :n_node]
            total += node_size

        #print(id(batch.x))
        return batch




    #def forward(self, x, pos, edge_index, batch, image_size):
    def forward(self, batch):
        # grid cluster
        #data_list = batch.to_data_list()
        #print(data_list)
        #clusters = self.grid_cluster(batch)
        #degs = self.degree(batch)

        batch_mask = batch.batch

        tl = []
        tr = []
        bl = []
        br = []
        tl_masks = []
        tr_masks = []
        bl_masks = []
        br_masks = []
        #for i, cluster, deg in zip(range(batch.num_graphs), clusters, degs):
        heihei = 0
        for i  in range(batch.num_graphs):
            #batch.x[batch_mask == i]
            #print(i, cluster.shape, deg.shape)
            data = batch[i]

            #print(data.x.max())




            image_size = data.image_size
            h, w = image_size[:2]

            h = int(math.ceil(h / 2))
            w = int(math.ceil(w / 2))
            cluster = grid_cluster(data.pos, torch.tensor([h, w]).to(data.x.device))

            deg = degree(data.edge_index[1])


            #tl_mask = (cluster == 0) & (deg == 8)
            #print(tl_mask.sum(), 0000)
            tl_mask = torch.logical_and(cluster == 0, deg == 8)
            #print(tl_mask.shape)
            #node_size = tl_mask.size(0)
            #a = batch.x[heihei: heihei + node_size]
            #print((data.x - a).sum())
            #heihei += node_size

            #print(tl_mask.shape, data.x.shape)
            tl_masks.append(tl_mask)
            #print(tl_mask.sum(), 111)
            tr_mask = torch.logical_and(cluster == 1, deg == 8)
            tr_masks.append(tr_mask)
            #print(tr_mask.shape)

            bl_mask = torch.logical_and(cluster == 2, deg == 8)
            bl_masks.append(bl_mask)
            #print(bl_mask.shape)

            br_mask = torch.logical_and(cluster == 3, deg == 8)
            br_masks.append(br_mask)
            #print(br_mask.shape, cluster.shape)
            #print(tl_mask.shape, tr_mask.shape)
            #print(tl_mask.sum(), 'sum')
            #print(cluster.sum())
            #print(tl_mask.sum())
            #print(data.x.shape, tl_mask.shape)
            #tl.append(batch[i].x.masked_select(tl_mask[:, None]))
            #tr.append(batch[i].x.masked_select(tr_mask[:, None]))
            #bl.append(batch[i].x.masked_select(bl_mask[:, None]))
            #br.append(batch[i].x.masked_select(br_mask[:, None]))
            #tmp = data.x[tl_mask]
            #print(tmp, data.x[tl_mask])
            #print(id(tmp), id(data.x[tl_mask]))
            tl.append(data.x[tl_mask])
            tr.append(data.x[tr_mask])
            bl.append(data.x[bl_mask])
            br.append(data.x[br_mask])
            #print(tl[-1], 2222222)
            #print(id(d), id(data.x[tl_mask]))
            #print(batch[i].x[br_mask].shape, 1111)
            #print(tl[-1].shape)
            #tl.append(batch.x[batch_mask == i])

        #tl, tl_mask, tl_num_nodes = self.batching(tl)
        #tr, tr_mask, tr_num_nodes = self.batching(tr)
        #bl, bl_mask, bl_num_nodes = self.batching(bl)
        #br, br_mask, br_num_nodes = self.batching(br)
        tl, tl_num_nodes = self.batching(tl)
        tr, tr_num_nodes = self.batching(tr)
        bl, bl_num_nodes = self.batching(bl)
        br, br_num_nodes = self.batching(br)

        #print(tl.shape, tl_mask.shape, tl.max())
        #print(tl_mask)
        #tl = self.trans_tl(tl.permute(1, 0, 2), src_key_padding_mask=tl_mask)
        tl = self.trans_tl(tl.permute(1, 0, 2))
        tl = tl.permute(1, 0, 2)
        #tr = self.trans_tr(tr.permute(1, 0, 2), src_key_padding_mask=~tr_mask)
        tr = self.trans_tr(tr.permute(1, 0, 2))
        tr = tr.permute(1, 0, 2)
        #bl = self.trans_bl(bl.permute(1, 0, 2), src_key_padding_mask=~bl_mask)
        bl = self.trans_bl(bl.permute(1, 0, 2))
        bl = bl.permute(1, 0, 2)
        #br = self.trans_br(br.permute(1, 0, 2), src_key_padding_mask=~br_mask)
        br = self.trans_br(br.permute(1, 0, 2))
        br = br.permute(1, 0, 2)

        # mask
        assert len(tl_masks) == len(tl_num_nodes)
        #print(id(batch), tl_num_nodes)
        #print(tl_masks)
        #for i, j in zip(tl_masks, tl_num_nodes):
            #print(i.shape, j)

        #print(id(batch.x))
        self.assign_data(batch, tl, tl_masks, tl_num_nodes)
        self.assign_data(batch, tr, tr_masks, tr_num_nodes)
        self.assign_data(batch, bl, bl_masks, bl_num_nodes)
        self.assign_data(batch, br, br_masks, br_num_nodes)
        #for idx, tl_m, tr_m, bl_m, br_m, n_node in enumerate(zip(tl_masks, tr_masks, bl_masks, br_masks, num_nodes)):
            #print(tl_m.sum(), n_node)
            #batch.x[tl_m] = tl[idx, :n_node]

        #for idx, tr_m in enumerate(zip(tr_masks, num_nodes)):
            #batch.x[tr_m] = tl[idx, :num]
        #print(tl.shape)
        #batch.x[]
        return batch



        # degree



#dim = 64
#encoder_layer = nn.TransformerEncoderLayer(
#            d_model=dim,
#            nhead=8,
#            dim_feedforward=dim * 4)
#net = SplitTrans(encoder_layer, dim)
#print(net    )

#class HatNet(nn.Module):
#    def __init__(self, in_channels, dim, out_channels, num_nodes=548):
#    #def __init__(self, in_channels, dim, out_channels, num_nodes=2109):
#        super().__init__()
#        #num_nodes = num_nodes
#        #print(num_nodes)
#        ratio = 0.5
#        num_nodes = int(ratio * num_nodes)
#        #print(num_nodes)
#        self.pos_emb1 = nn.Parameter(torch.randn(1, num_nodes, dim))
#
#        encoder_layer = nn.TransformerEncoderLayer(
#            d_model=dim,
#            #nhead=2,
#            nhead=8,
#            dim_feedforward=dim * 4)
#            #dim_feedforward=dim)
#
#        self.pool1 = nn.Linear(dim, num_nodes)
#        num_nodes = int(ratio * num_nodes)
#        self.pos_emb2 = nn.Parameter(torch.randn(1, num_nodes, dim))
#        #print(num_nodes)
#        #import sys; sys.exit()
#        self.pool2 = nn.Linear(dim, num_nodes)
#
#        #self.split_trans = SplitTrans(encoder_layer, dim)
#
#        #self.conv1 = GINConv(
#        #    nn.Sequential(nn.Linear(in_channels, dim), nn.BatchNorm1d(dim), nn.ReLU(),
#        #               nn.Linear(dim, dim), nn.ReLU()))
#        self.conv1 = GINConv(NN(in_channels, dim))
#        self.conv2 = GINConv(NN(dim, dim))
#        self.conv3 = DenseGINConv(NN(dim, dim))
#        self.conv4 = DenseGINConv(NN(dim, dim))
#        self.conv5 = DenseGINConv(NN(dim, dim))
#
#        self.trans1 = nn.TransformerEncoder(encoder_layer, num_layers=6)
#        self.trans2 = nn.TransformerEncoder(encoder_layer, num_layers=6)
#        #self.conv2 = GINConv(
#        #    nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
#        #               nn.Linear(dim, dim), nn.ReLU()))
#
#        #self.conv3 = DenseGINConv(
#        #    nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
#        #               nn.Linear(dim, dim), nn.ReLU()))
#
#        #self.conv4 = DenseGINConv(
#        #    nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
#        #               nn.Linear(dim, dim), nn.ReLU()))
#
#        #self.conv5 = DenseGINConv(
#        #    nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
#        #               nn.Linear(dim, dim), nn.ReLU()))
#
#        #self.trans1 = nn.Transformer(d_model=dim, num_decoder_layers=0, num_encoder_layers=6)
#        #self.trans2 = nn.Transformer(d_model=dim, num_decoder_layers=0, num_encoder_layers=6)
#        self.fc1 = nn.Sequential(nn.Linear(dim * 21, dim * 21 // 4), nn.BatchNorm1d(dim * 21 // 4), nn.ReLU())
#        #self.fc1 = nn.Sequential(nn.Linear(dim * 15, dim * 15 // 4), nn.BatchNorm1d(dim * 15 // 4), nn.ReLU())
#        self.dropout = nn.Dropout()
#        #self.fc2 = nn.Linear(dim * 15 // 4, out_channels)
#        self.fc2 = nn.Linear(dim * 21 // 4, out_channels)
#
#    #def test_deg(self, data):
#        #for i in data:
#            #print(i)
#
#        #for i in range(data.num_graphs):
#        #    deg = degree(data.edge_index[0], data.num_nodes)
#        #    print(deg.max())
#
#
#    def forward(self, data):
#        #self.test_deg(data)
#        #deg = degree(data.edge_index[0], data.num_nodes)
#        #print(deg.max(), 222222222)
#        #x, edge_index, batch = data.x, data.edge_index, data.batch
#        #print(data.edge_index[0].shape, data.num_nodes)
#
#        outputs = []
#        data.x = self.conv1(data.x, data.edge_index)
#        outputs.append(global_add_pool(data.x, data.batch))
#        outputs.append(global_mean_pool(data.x, data.batch))
#        outputs.append(global_max_pool(data.x, data.batch))
#
#        data.x = self.conv2(data.x, data.edge_index)
#        outputs.append(global_add_pool(data.x, data.batch))
#        outputs.append(global_mean_pool(data.x, data.batch))
#        outputs.append(global_max_pool(data.x, data.batch))
#
#        x, edge_index, batch = data.x, data.edge_index, data.batch
#
#        x, mask = to_dense_batch(x, batch)
#        adj = to_dense_adj(edge_index, batch)
#
#        s = self.pool1(x)
#        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)
#        outputs.append(torch.mean(x, dim=1))
#        outputs.append(torch.sum(x, dim=1))
#        outputs.append(torch.max(x, dim=1)[0])
#
#        x = self.conv3(x, adj)
#        outputs.append(torch.mean(x, dim=1))
#        outputs.append(torch.sum(x, dim=1))
#        outputs.append(torch.max(x, dim=1)[0])
#
#        x = self.conv4(x, adj)
#        x += self.pos_emb1
#        x = x.permute(1, 0, 2)
#        x = self.trans1(x)
#        x = x.permute(1, 0, 2)
#
#        outputs.append(torch.mean(x, dim=1))
#        outputs.append(torch.sum(x, dim=1))
#        outputs.append(torch.max(x, dim=1)[0])
#
#
#        s = self.pool2(x)
#        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
#        #print(x.shape) # 267
#        outputs.append(torch.mean(x, dim=1))
#        outputs.append(torch.sum(x, dim=1))
#        outputs.append(torch.max(x, dim=1)[0])
#
#        x = self.conv5(x, adj)
#        x += self.pos_emb2
#        x = x.permute(1, 0, 2)
#        #print(x.shape)
#        x = self.trans2(x)
#        x = x.permute(1, 0, 2)
#        #print(x.shape)
#        #import sys; sys.exit()
#        outputs.append(torch.mean(x, dim=1))
#        outputs.append(torch.sum(x, dim=1))
#        outputs.append(torch.max(x, dim=1)[0])
#
#        x = torch.cat(outputs, dim=1)
#        #print(x.shape, self.fc1)
#        #print(x.shape, self.fc1)
#        x = self.fc1(x)
#        x = self.dropout(x)
#        x = self.fc2(x)
#        mc_loss = mc1 + mc2
#        o_loss = o1 + o2
#        loss = F.cross_entropy(x, data.y, size_average=True) + mc_loss + o_loss
#        return x, loss


# mincut pool
class HatNet(nn.Module):
    def __init__(self, in_channels, dim, out_channels, num_nodes=548):
    #def __init__(self, in_channels, dim, out_channels, num_nodes=2109):
        super().__init__()
        #num_nodes = num_nodes
        #print(num_nodes)
        ratio = 0.5
        num_nodes = int(ratio * num_nodes)
        print(num_nodes)
        self.pos_emb1 = nn.Parameter(torch.randn(1, num_nodes, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=2,
            #nhead=8,
            #dim_feedforward=dim * 4)
            #dim_feedforward=2048)
            #dim_feedforward=dim)
            #dim_feedforward=dim)
        )

        self.pool1 = nn.Linear(dim, num_nodes)
        num_nodes = int(ratio * num_nodes)
        self.pos_emb2 = nn.Parameter(torch.randn(1, num_nodes, dim))
        #print(num_nodes)
        #import sys; sys.exit()
        self.pool2 = nn.Linear(dim, num_nodes)

        #self.split_trans = SplitTrans(encoder_layer, dim)

        #self.conv1 = GINConv(
        #    nn.Sequential(nn.Linear(in_channels, dim), nn.BatchNorm1d(dim), nn.ReLU(),
        #               nn.Linear(dim, dim), nn.ReLU()))
        self.conv1 = GINConv(NN(in_channels, dim))
        self.conv2 = GINConv(NN(dim, dim))
        self.conv3 = DenseGINConv(NN(dim, dim))
        self.conv4 = DenseGINConv(NN(dim, dim))
        self.conv5 = DenseGINConv(NN(dim, dim))

        self.trans1 = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.trans2 = nn.TransformerEncoder(encoder_layer, num_layers=6)
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

        #self.trans1 = nn.Transformer(d_model=dim, num_decoder_layers=0, num_encoder_layers=6)
        #self.trans2 = nn.Transformer(d_model=dim, num_decoder_layers=0, num_encoder_layers=6)
        #ratio = 21
        ratio = 21
        self.fc1 = nn.Sequential(nn.Linear(dim * ratio, dim * ratio // 4), nn.BatchNorm1d(dim * ratio // 4), nn.ReLU())
        #self.fc1 = nn.Sequential(nn.Linear(dim * 15, dim * 15 // 4), nn.BatchNorm1d(dim * 15 // 4), nn.ReLU())
        self.dropout = nn.Dropout()
        #self.fc2 = nn.Linear(dim * 15 // 4, out_channels)
        self.fc2 = nn.Linear(dim * ratio // 4, out_channels)

    #def test_deg(self, data):
        #for i in data:
            #print(i)

        #for i in range(data.num_graphs):
        #    deg = degree(data.edge_index[0], data.num_nodes)
        #    print(deg.max())

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def forward(self, data):
        #self.test_deg(data)
        #deg = degree(data.edge_index[0], data.num_nodes)
        #print(deg.max(), 222222222)
        #x, edge_index, batch = data.x, data.edge_index, data.batch
        #print(data.edge_index[0].shape, data.num_nodes)

        outputs = []
        data.x = self.conv1(data.x, data.edge_index)
        outputs.append(global_add_pool(data.x, data.batch))
        outputs.append(global_mean_pool(data.x, data.batch))
        outputs.append(global_max_pool(data.x, data.batch))

        data.x = self.conv2(data.x, data.edge_index)
        outputs.append(global_add_pool(data.x, data.batch))
        outputs.append(global_mean_pool(data.x, data.batch))
        outputs.append(global_max_pool(data.x, data.batch))

        x, edge_index, batch = data.x, data.edge_index, data.batch

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
        #x += self.pos_emb1
        x = x.permute(1, 0, 2)
        x = self.trans1(x)
        x = x.permute(1, 0, 2)

        outputs.append(torch.mean(x, dim=1))
        outputs.append(torch.sum(x, dim=1))
        outputs.append(torch.max(x, dim=1)[0])


        s = self.pool2(x)
        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
        #print(x.shape) # 267
        outputs.append(torch.mean(x, dim=1))
        outputs.append(torch.sum(x, dim=1))
        outputs.append(torch.max(x, dim=1)[0])

        x = self.conv5(x, adj)
        #x += self.pos_emb2
        x = x.permute(1, 0, 2)
        #print(x.shape)
        x = self.trans2(x)
        x = x.permute(1, 0, 2)
        #print(x.shape)
        #import sys; sys.exit()
        outputs.append(torch.mean(x, dim=1))
        outputs.append(torch.sum(x, dim=1))
        outputs.append(torch.max(x, dim=1)[0])

        x = torch.cat(outputs, dim=1)
        #print(x.shape, self.fc1)
        #print(x.shape, self.fc1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        mc_loss = mc1 + mc2
        o_loss = o1 + o2
        loss = F.cross_entropy(x, data.y, size_average=True) + mc_loss + o_loss
        return x, loss


## weakly supervised

class Prostate(nn.Module):
    def __init__(self, in_channels, dim, out_channels):
        super().__init__()
        self.conv = SAGEConv(in_channels, dim)
        self.conv1 = SAGEConv(dim, dim)
        self.conv2 = SAGEConv(dim, dim)
        #self.conv2 = SAGEConv(dim, dim, num_layers=2)
        self.pool1 = SAGPooling(dim, min_score=0.001, GNN=GCNConv)
        self.conv3 = SAGEConv(dim, dim)
        self.conv4 = SAGEConv(dim, dim)
        self.pool2 = SAGPooling(dim, min_score=0.001, GNN=GCNConv)
        self.fc = torch.nn.Linear(dim, out_channels)
        self.fc1 = nn.Sequential(nn.Linear(dim * 4, dim * 4), nn.BatchNorm1d(dim * 4), nn.ReLU())
        #self.fc1 = nn.Sequential(nn.Linear(dim * 15, dim * 15 // 4), nn.BatchNorm1d(dim * 15 // 4), nn.ReLU())
        self.dropout = nn.Dropout()
        #self.fc2 = nn.Linear(dim * 15 // 4, out_channels)
        self.fc2 = nn.Linear(dim * 4, out_channels)

    def forward(self, data):
        outputs = []
        x, edge_index, batch = data.x, data.edge_index, data.batch

        #print(x.shape)
        x = self.conv(x, edge_index)
        #outputs.append(torch.max(x, dim=0)[0])
        #print(x.shape)
        outputs.append(global_max_pool(x, batch))

        x = self.conv1(x, edge_index)
        #print(x.shape)
        #outputs.append(torch.max(x, dim=0)[0])
        outputs.append(global_max_pool(x, batch))
        #print(x.shape)
        #x, edge_index, _, batch, perm, score = self.pool1(
        #    x, edge_index, None, batch)

        x = self.conv3(x, edge_index)
        #print(x.shape)
        #outputs.append(torch.max(x, dim=0)[0])
        outputs.append(global_max_pool(x, batch))
        #x = self.conv4(x, edge_index)
        x, edge_index, _, batch, perm, score = self.pool2(
            x, edge_index, None, batch)

        outputs.append(global_max_pool(x, batch))
        #outputs.append(torch.max(x, dim=0)[0])
        #x = global_mean_pool(x, batch)
        #x = global_max_pool(x, batch)
        #print(x.shape)
        #outputs.append(torch.max(x, dim=0)[0])
        #outputs.append(x)
        #print()
        #for i in outputs:
        #    print(i.shape)

        x = torch.cat(outputs, dim=1)
        #print(x.shape)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        loss = F.cross_entropy(x, data.y, size_average=True)
        return x, loss





# self attention pool
#class HatNet(nn.Module):
#    def __init__(self, in_channels, dim, out_channels, num_nodes=548):
#        super().__init__()
#        #num_nodes = num_nodes
#        #num_nodes = int(0.5 * num_nodes)
#        #self.pool1 = nn.Linear(dim, num_nodes)
#        #num_nodes = int(0.5 * num_nodes)
#        #self.pool2 = nn.Linear(dim, num_nodes)
#
#        #self.conv1 = GINConv(
#        #    nn.Sequential(nn.Linear(in_channels, dim), nn.BatchNorm1d(dim), nn.ReLU(),
#        #               nn.Linear(dim, dim), nn.ReLU()))
#        self.conv1 = GINConv(NN(in_channels, dim))
#        self.conv2 = GINConv(NN(dim, dim))
#        self.conv3 = GINConv(NN(dim, dim))
#        self.conv4 = GINConv(NN(dim, dim))
#        self.conv5 = GINConv(NN(dim, dim))
#        self.pool1 = SAGPooling(dim, GNN=GCNConv)
#        self.pool2 = SAGPooling(dim, GNN=GCNConv)
#        #self.conv3 = DenseGINConv(NN(dim, dim))
#        #self.conv4 = DenseGINConv(NN(dim, dim))
#        #self.conv5 = DenseGINConv(NN(dim, dim))
#
#        #self.conv2 = GINConv(
#        #    nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
#        #               nn.Linear(dim, dim), nn.ReLU()))
#
#        #self.conv3 = DenseGINConv(
#        #    nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
#        #               nn.Linear(dim, dim), nn.ReLU()))
#
#        #self.conv4 = DenseGINConv(
#        #    nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
#        #               nn.Linear(dim, dim), nn.ReLU()))
#
#        #self.conv5 = DenseGINConv(
#        #    nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
#        #               nn.Linear(dim, dim), nn.ReLU()))
#
#        self.fc1 = nn.Sequential(nn.Linear(dim * 21, dim * 21 // 4), nn.BatchNorm1d(dim * 21 // 4), nn.ReLU())
#        self.dropout = nn.Dropout()
#        self.fc2 = nn.Linear(dim * 21 // 4, out_channels)
#
#    def forward(self, data):
#        x, edge_index, batch = data.x, data.edge_index, data.batch
#
#        outputs = []
#        x = self.conv1(x, edge_index)
#        outputs.append(global_add_pool(x, batch))
#        outputs.append(global_mean_pool(x, batch))
#        outputs.append(global_max_pool(x, batch))
#
#        x = self.conv2(x, edge_index)
#        outputs.append(global_add_pool(x, batch))
#        outputs.append(global_mean_pool(x, batch))
#        outputs.append(global_max_pool(x, batch))
#
#        #x, mask = to_dense_batch(x, batch)
#        #adj = to_dense_adj(edge_index, batch)
#        #s = self.pool1(x)
#        #print(x.shape)
#        #x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)
#        x, edge_index, _, batch, perm, score = self.pool1(
#            x, edge_index, None, batch)
#        #print(x.shape)
#        outputs.append(global_add_pool(x, batch))
#        outputs.append(global_mean_pool(x, batch))
#        outputs.append(global_max_pool(x, batch))
#        #outputs.append(torch.mean(x, dim=1))
#        #outputs.append(torch.sum(x, dim=1))
#        #outputs.append(torch.max(x, dim=1)[0])
#
#        x = self.conv3(x, edge_index)
#        outputs.append(global_add_pool(x, batch))
#        outputs.append(global_mean_pool(x, batch))
#        outputs.append(global_max_pool(x, batch))
#        #outputs.append(torch.mean(x, dim=1))
#        #outputs.append(torch.sum(x, dim=1))
#        #outputs.append(torch.max(x, dim=1)[0])
#
#        x = self.conv4(x, edge_index)
#        outputs.append(global_add_pool(x, batch))
#        outputs.append(global_mean_pool(x, batch))
#        outputs.append(global_max_pool(x, batch))
#        #outputs.append(torch.mean(x, dim=1))
#        #outputs.append(torch.sum(x, dim=1))
#        #outputs.append(torch.max(x, dim=1)[0])
#
#        #s = self.pool2(x)
#        x, edge_index, _, batch, perm, score = self.pool2(
#            x, edge_index, None, batch)
#
#        #print(x.shape)
#        #x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
#        outputs.append(global_add_pool(x, batch))
#        outputs.append(global_mean_pool(x, batch))
#        outputs.append(global_max_pool(x, batch))
#        #outputs.append(torch.mean(x, dim=1))
#        #outputs.append(torch.sum(x, dim=1))
#        #outputs.append(torch.max(x, dim=1)[0])
#
#        x = self.conv5(x, edge_index)
#        outputs.append(global_add_pool(x, batch))
#        outputs.append(global_mean_pool(x, batch))
#        outputs.append(global_max_pool(x, batch))
#        #outputs.append(torch.mean(x, dim=1))
#        #outputs.append(torch.sum(x, dim=1))
#        #outputs.append(torch.max(x, dim=1)[0])
#
#
#        x = torch.cat(outputs, dim=1)
#        #print(x.shape)
#        x = self.fc1(x)
#        x = self.dropout(x)
#        x = self.fc2(x)
#        loss = F.cross_entropy(x, data.y, size_average=True)
#        #print(x.shape, loss)
#        #import sys; sys.exit()
#        return x, loss
#
class HatNetC(nn.Module):
    def __init__(self, in_channels, dim, out_channels, num_nodes=548):
    #def __init__(self, in_channels, dim, out_channels, num_nodes=2109):
        super().__init__()
        #num_nodes = num_nodes
        #print(num_nodes)
        ratio = 0.5
        num_nodes = int(ratio * num_nodes)
        print(num_nodes)
        #self.pos_emb1 = nn.Parameter(torch.randn(1, num_nodes, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=2,
            #nhead=8,
            #dim_feedforward=dim * 4)
            #dim_feedforward=2048)
            #dim_feedforward=dim)
            dim_feedforward=dim)

        self.pool1 = nn.Linear(dim, num_nodes)
        num_nodes = int(ratio * num_nodes)
        #self.pos_emb2 = nn.Parameter(torch.randn(1, num_nodes, dim))
        #print(num_nodes)
        #import sys; sys.exit()
        #num_nodes = int(ratio * num_nodes)
        self.pool2 = nn.Linear(dim, num_nodes)

        num_nodes = int(ratio * num_nodes)
        self.pool3 = nn.Linear(dim, num_nodes)
        #self.split_trans = SplitTrans(encoder_layer, dim)

        #self.conv1 = GINConv(
        #    nn.Sequential(nn.Linear(in_channels, dim), nn.BatchNorm1d(dim), nn.ReLU(),
        #               nn.Linear(dim, dim), nn.ReLU()))
        self.conv1 = GINConv(NN(in_channels, dim))
        #self.conv2 = GINConv(NN(dim, dim))
        self.conv2 = DenseGINConv(NN(dim, dim))
        self.conv3 = DenseGINConv(NN(dim, dim))
        #self.conv5 = DenseGINConv(NN(dim, dim))

        self.trans1 = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.trans2 = nn.TransformerEncoder(encoder_layer, num_layers=6)
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

        #self.trans1 = nn.Transformer(d_model=dim, num_decoder_layers=0, num_encoder_layers=6)
        #self.trans2 = nn.Transformer(d_model=dim, num_decoder_layers=0, num_encoder_layers=6)
        #ratio = 21
        ratio = 3
        self.fc1 = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU())
        #self.fc1 = nn.Sequential(nn.Linear(dim * 15, dim * 15 // 4), nn.BatchNorm1d(dim * 15 // 4), nn.ReLU())
        self.dropout = nn.Dropout()
        #self.fc2 = nn.Linear(dim * 15 // 4, out_channels)
        self.fc2 = nn.Linear(dim , out_channels)
        self.jk = JumpingKnowledge('lstm', dim, 3)

    #def test_deg(self, data):
        #for i in data:
            #print(i)

        #for i in range(data.num_graphs):
        #    deg = degree(data.edge_index[0], data.num_nodes)
        #    print(deg.max())

    #def on_load_checkpoint(self, checkpoint: dict) -> None:
    #    state_dict = checkpoint["state_dict"]
    #    model_state_dict = self.state_dict()
    #    is_changed = False
    #    for k in state_dict:
    #        if k in model_state_dict:
    #            if state_dict[k].shape != model_state_dict[k].shape:
    #                print(f"Skip loading parameter: {k}, "
    #                            f"required shape: {model_state_dict[k].shape}, "
    #                            f"loaded shape: {state_dict[k].shape}")
    #                state_dict[k] = model_state_dict[k]
    #                is_changed = True
    #        else:
    #            print(f"Dropping parameter {k}")
    #            is_changed = True

    #    if is_changed:
    #        checkpoint.pop("optimizer_states", None)

    def forward(self, data):
        #self.test_deg(data)
        #deg = degree(data.edge_index[0], data.num_nodes)
        #print(deg.max(), 222222222)
        #x, edge_index, batch = data.x, data.edge_index, data.batch
        #print(data.edge_index[0].shape, data.num_nodes)

        outputs = []
        # stage 1
        data.x = self.conv1(data.x, data.edge_index)
        #outputs.append(global_add_pool(data.x, data.batch))
        #outputs.append(global_mean_pool(data.x, data.batch))
        #outputs.append(global_max_pool(data.x, data.batch))

        #data.x = self.conv2(data.x, data.edge_index)
        #outputs.append(global_add_pool(data.x, data.batch))
        #outputs.append(global_mean_pool(data.x, data.batch))
        #outputs.append(global_max_pool(data.x, data.batch))

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)

        s = self.pool1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)
        #outputs.append(torch.mean(x, dim=1))
        #outputs.append(torch.sum(x, dim=1))
        outputs.append(torch.max(x, dim=1)[0])


        # stage 2 ####################################

        x = self.conv2(x, adj)
        s = self.pool2(x)
        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
        x = x.permute(1, 0, 2)
        x = self.trans1(x)
        x = x.permute(1, 0, 2)
        #outputs.append(torch.mean(x, dim=1))
        #outputs.append(torch.sum(x, dim=1))
        outputs.append(torch.max(x, dim=1)[0])
        # end stage 2 ####################################

        # stage 3 ####################################
        x = self.conv3(x, adj)
        s = self.pool3(x)
        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
        x = x.permute(1, 0, 2)
        x = self.trans1(x)
        x = x.permute(1, 0, 2)

        #x = self.conv4(x, adj)
        ##x += self.pos_emb1
        #x = x.permute(1, 0, 2)
        #x = self.trans1(x)
        #x = x.permute(1, 0, 2)

        #outputs.append(torch.mean(x, dim=1))
        #outputs.append(torch.sum(x, dim=1))
        outputs.append(torch.max(x, dim=1)[0])
        # end stage 3 ####################################


        #s = self.pool2(x)
        #x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
        ##print(x.shape) # 267
        ##outputs.append(torch.mean(x, dim=1))
        ##outputs.append(torch.sum(x, dim=1))
        ##outputs.append(torch.max(x, dim=1)[0])

        #x = self.conv5(x, adj)
        ##x += self.pos_emb2
        #x = x.permute(1, 0, 2)
        ##print(x.shape)
        #x = self.trans2(x)
        #x = x.permute(1, 0, 2)
        #print(x.shape)
        #import sys; sys.exit()
        #outputs.append(torch.mean(x, dim=1))
        #outputs.append(torch.sum(x, dim=1))
        #outputs.append(torch.max(x, dim=1)[0])

        #x = torch.cat(outputs, dim=1)
        x = self.jk(outputs)
        #print(x.shape)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        mc_loss = mc1 + mc2
        o_loss = o1 + o2
        loss = F.cross_entropy(x, data.y, size_average=True) + mc_loss + o_loss
        return x, loss