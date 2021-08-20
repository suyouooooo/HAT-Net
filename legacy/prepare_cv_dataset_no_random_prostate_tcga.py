
import os.path as osp
import os
from pathlib import Path

import sys
from pathlib import Path
from functools import partial
sys.path.append(os.getcwd())
import torch
from multiprocessing import Pool
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
#from torch_geometric.nn.pool import avg_pool

import copy
import random
import numpy as np
import glob
from common.utils import mkdirs,FarthestSampler,filter_sampled_indice, mkdir

from setting import CrossValidSetting
from dataflow.graph_sampler import random_sample_graph2


from torch_cluster import grid_cluster
from torch_scatter import scatter
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos

from stich import LMDBFolder, LMDBDataset

def _avg_pool_x(cluster, x, size=None):
    return scatter(x, cluster, dim=0, dim_size=size, reduce='mean')

def avg_pool(cluster, data):
    cluster, perm = consecutive_cluster(cluster)
    x = None if data.x is None else _avg_pool_x(cluster, data.x)
    pos = None if data.pos is None else pool_pos(cluster, data.pos)
    data.x = x
    data.pos = pos

    return data

def _add_pool_x(cluster, x, size=None):
    return scatter(x, cluster, dim=0, dim_size=size, reduce='add')

def add_pool(cluster, data):
    cluster, perm = consecutive_cluster(cluster)
    x = None if data.x is None else _add_pool_x(cluster, data.x)
    pos = None if data.pos is None else pool_pos(cluster, data.pos)
    data.x = x
    data.x[:, -2:] = pos
    data.pos = pos

    return data

def read_data(dataset, raw_fp):
    res = dataset.get_file_by_path(raw_fp)
    feats = res['feat']
    coords = res['coord']

    feats = np.concatenate((feats, coords), axis= -1)
    coords = torch.from_numpy(coords).to(torch.float)
    feats = torch.from_numpy(feats).to(torch.float)

    if '1_normal' in  raw_fp:
        label = 0
    elif '2_low_grade' in raw_fp:
        label = 1
    elif '3_high_grade' in raw_fp:
        label = 2
    else:
        raise ValueError('value error')

    y = torch.tensor([label], dtype=torch.long)
    data = Data(x=feats, pos=coords, y=y)

    return data

def gen(raw_path, dataset, max_neighbours, epoch, sample_method):
    # important: to select sample method, change sample_method in both main function and def gen
    #max_neighbours = 8
    #epoch = 1
    #graph_sampler = 'knn'
    #mask = 'cia'
    ##sample_method= 'fuse'
    #sample_method= 'avg'
    setting = CrossValidSetting()
    processed_dir = os.path.join(setting.root, 'proto',
                                 'fix_%s_%s_%s' % (sample_method, 'cia', 'knn'))

     # Read data from `raw_path`
     ## raw_path is /data/hdd1/syh/PycharmProjects/CGC-Net/data/raw/CRC/fold_1/1_normal/xx.png
     # raw_path is /data/hdd1/syh/PycharmProjects/CGC-Net/data/proto/feature/CRC/fold_1/1_normal/xx(无后缀)
    data = read_data(dataset, raw_path)
     # sample epoch time
    num_nodes = data.x.shape[0]
    if num_nodes == 0:
        return
    num_sample = num_nodes
    for i in range(epoch):
       subdata = copy.deepcopy(data)
       clusters = grid_cluster(subdata.pos, torch.Tensor([64, 64]))
       subdata = avg_pool(clusters, subdata)

       # generate the graph
       #if graph_sampler == 'knn':
       edge_index = radius_graph(subdata.pos, 100, None, True, max_neighbours)
       #else:
       #    edge_index = random_sample_graph2(choice, distance, 100, True,
                                     #n_sample=8,sparse=True)
       subdata.edge_index=edge_index

       raw_path1 = Path(raw_path)
       parts = raw_path1.parts
       save_fp = os.path.join(processed_dir, str(i), parts[0], parts[2].replace('.npy', '.pt'))
       print(save_fp)
       #torch.save(subdata, save_fp)
       #print(osp.join(processed_dir,str(i),
       #                          raw_path.split('/')[-3],
       #                             raw_path.split('/')[-1].split('.')[0] + '.pt'))



       #print('before', data.x.shape, 'after:', subdata.x.shape)
       #torch.save(subdata, osp.join(processed_dir,str(i),
       #                          raw_path.split('/')[-3],
       #                             raw_path.split('/')[-1].split('.')[0] + '.pt'))

if __name__ == '__main__':
    # important: to select sample method, change sample_method in both main function and def gen
    setting = CrossValidSetting()
    folds = ['fold_1', 'fold_2', 'fold_3']
    #sample_method = 'fuse'
    sample_method = 'avg'
    mask = 'cia'
    graph_sampler = 'knn'
    epoch = 1
    #sampler = FarthestSampler()
    print(setting.root)

    processed_dir = os.path.join(setting.root, 'proto',
                                      'fix_%s_%s_%s' % (sample_method, mask, graph_sampler))

    for f in folds:
        for j in range(epoch):
            #print(epoch, j)
            # print('path is : ' + str(osp.join(processed_dir, '%d' % j, f)))
            mkdirs(osp.join(processed_dir, '%d' % j, f))

    # print("original_files : " + str(original_files))
    lmdb_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_lmdb/extended_crc/feat/'
    #lmdb_folder = LMDBFolder('/data/smb/syh/PycharmProjects/CGC-Net/data_lmdb/extended_crc/feat/')
    lmdb_folder = LMDBFolder(lmdb_path)
    lmdb_dataset = LMDBDataset(lmdb_folder, 224 * 16)
    file_path_lists = lmdb_dataset.file_path_lists
    gen = partial(gen, dataset=lmdb_dataset, max_neighbours=8, sample_method='avg', epoch=1)
    for file_path in file_path_lists:
        gen(file_path)



    #res = []
    #for file_fp in file_path_lists:
    #    full_path = Path(str(file_fp))
    #    print(full_path)
    #    res.append(full_path.relative_to(lmdb_path))




    #p = Pool(6)
    #print(len(file_path_lists))
    #sys.exit()
    #arr = p.map(gen, file_path_lists)
    #p.close()
    #p.join()
    # gen('/data/hdd1/syh/PycharmProjects/CGC-Net/data/proto/feature/CRC/fold_1/1_normal/Patient_013_03_Normal')