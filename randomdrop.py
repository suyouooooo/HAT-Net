import glob
import os

import torch
import torch_geometric
from torch_geometric.nn import radius_graph




def dropnodes(data, ratio):
    length = len(data.x)
    sample = torch.rand(length).topk(int(length * (1 - ratio))).indices
    mask = torch.zeros(length, dtype=torch.bool)
    mask.scatter_(dim=0, index=sample, value=True)

    data.x = data.x[mask]
    data.pos = data.pos[mask]
    #print('after', data)
    #data = Data(x=feat, pos=coord, y=y)
    edge_index = radius_graph(data.pos, 100, None, True, 8)

    data.edge_index = edge_index
    return data




path = '/data/smb/syh/PycharmProjects/CGC-Net/data_res50/proto/fix_fuse_cia_knn/0/'
save_path50 = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/drop50percentnodes'
save_path30 = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/drop30percentnodes'
save_path10 = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/drop10percentnodes'
def dropgraph(path, ratio):
    nodes_count = 0
    graph_count = 0
    import time
    start = time.time()
    for idx, pt_fp in enumerate(glob.iglob(os.path.join(path, '**', '*.pt'), recursive=True)):
        data = torch.load(pt_fp)
        nodes_count += len(data.x)
        graph_count += 1
        pt_fp = os.path.normpath(pt_fp)
        pt_fp = pt_fp.split(os.sep)

        # 50
        data50 = dropnodes(data.clone(), 0.5)
        save_fp = os.path.join(save_path50, pt_fp[-2], pt_fp[-1])
        #print(save_fp)
        os.makedirs(os.path.dirname(save_fp), exist_ok=True)
        torch.save(data50, save_fp)


        # 30
        data30 = dropnodes(data.clone(), 0.3)
        save_fp = os.path.join(save_path30, pt_fp[-2], pt_fp[-1])
        #print(save_fp)
        os.makedirs(os.path.dirname(save_fp), exist_ok=True)
        torch.save(data30, save_fp)


        # 10
        data10 = dropnodes(data.clone(), 0.1)
        save_fp = os.path.join(save_path10, pt_fp[-2], pt_fp[-1])
        #print(save_fp)
        os.makedirs(os.path.dirname(save_fp), exist_ok=True)
        torch.save(data10, save_fp)

        #print(data, data10, data30, data50)

        if idx % 500 == 0:
            print(idx)
            finish = time.time()
            print(nodes_count / graph_count,  graph_count / (finish - start))

dropgraph(path, 0.5)