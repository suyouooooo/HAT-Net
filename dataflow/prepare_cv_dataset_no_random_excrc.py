
import os.path as osp
import os
from pathlib import Path

import sys
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
    #print(data.x.shape)
    #print(data.x[:, -2:])
    #print(data.pos[:, -2:])
    #print(data.x[:,-2].shape)
    data.x[:, -2:] = pos
    data.pos = pos
    #print(data.x[:, -2:])
    #print(pos.shape)

    return data

#def avg_coord(coordinates, size):
#    clusters = grid_cluster(coordinates, torch.Tensor([size, size]))
#    #tmp1 = clusters
#    #print(clusters)
#    clusters, perm = consecutive_cluster(clusters)
#    pos =  pool_pos(clusters, coordinates)
#
#    #image = cv2.imread('/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC/fold_2/3_high_grade/H09-18586_A2H_E_1_4_grade_3_2017_3585_180.png')
#    #visulaize_cluster(image, clusters, pos)
#    #import sys; sys.exit()
#    return pos




#def _read_one_raw_graph( raw_file_path):
#    # import pdb;pdb.set_trace()
#    nodes_features = np.load( raw_file_path + '.npy')
#    coordinates = np.load(raw_file_path.replace('feature', 'coordinate') + '.npy')
#    nodes_features = np.concatenate((nodes_features, coordinates), axis= -1)
#    coordinates = torch.from_numpy(coordinates).to(torch.float)
#    nodes_features = torch.from_numpy(nodes_features ).to(torch.float)
#    if '1_normal' in raw_file_path:
#        label = 0
#    elif '2_low_grade' in raw_file_path:
#        label = 1
#    else:
#        label = 2
#    y = torch.tensor([label], dtype=torch.long)
#    data = Data(x = nodes_features, pos = coordinates, y = y)
#    return data
#def _read_one_raw_graph( raw_file_path):
#    # import pdb;pdb.set_trace()
#
#    nodes_features =
#    #nodes_features = np.load( raw_file_path + '.npy')
#    #coordinates = np.load(raw_file_path.replace('feature', 'coordinate') + '.npy')
#    nodes_features = np.concatenate((nodes_features, coordinates), axis= -1)
#    coordinates = torch.from_numpy(coordinates).to(torch.float)
#    nodes_features = torch.from_numpy(nodes_features ).to(torch.float)
#    if '1_normal' in raw_file_path:
#        label = 0
#    elif '2_low_grade' in raw_file_path:
#        label = 1
#    else:
#        label = 2
#    y = torch.tensor([label], dtype=torch.long)
#    data = Data(x = nodes_features, pos = coordinates, y = y)

#def _read_one_raw_graph(raw_file_path)

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

def gen(raw_path, dataset):
    # important: to select sample method, change sample_method in both main function and def gen
    max_neighbours = 8
    epoch = 1
    graph_sampler = 'knn'
    mask = 'cia'
    #sample_method= 'fuse'
    sample_method= 'avg'
    setting = CrossValidSetting()
    processed_dir = os.path.join(setting.root, 'proto',
                                 'fix_%s_%s_%s' % (sample_method, mask, graph_sampler))

     # Read data from `raw_path`
     ## raw_path is /data/hdd1/syh/PycharmProjects/CGC-Net/data/raw/CRC/fold_1/1_normal/xx.png
     # raw_path is /data/hdd1/syh/PycharmProjects/CGC-Net/data/proto/feature/CRC/fold_1/1_normal/xx(无后缀)
    #print(raw_path)
    #print(raw_path)
    data = read_data(dataset, raw_path)
     # sample epoch time
    num_nodes = data.x.shape[0]
    if num_nodes == 0:
        #print(raw_path)
        return
    #print('before', data.x.shape)
    num_sample = num_nodes
    #distance_path = os.path.join(setting.root, 'proto', 'distance', 'CRC',
                                  #raw_path.split('/')[-3], raw_path.split('/')[-2], raw_path.split('/')[-1] + '.pt')
    #distance = np.load(distance_path.replace('.pt', '.npy'))
    for i in range(epoch):
       subdata = copy.deepcopy(data)
       #choice, num_subsample = _sampling(num_sample,0.5,distance)
       #for key, item in subdata:
       #    if torch.is_tensor(item) and item.size(0) == num_nodes:
       #        subdata[key] = item[choice]
       clusters = grid_cluster(subdata.pos, torch.Tensor([64, 64]))
       subdata = avg_pool(clusters, subdata)
       #subdata = add_pool(clusters, subdata)

       # generate the graph
       if graph_sampler == 'knn':
           edge_index = radius_graph(subdata.pos, 100, None, True, max_neighbours)
       else:
           edge_index = random_sample_graph2(choice, distance, 100, True,
                                     n_sample=8,sparse=True)
       subdata.edge_index=edge_index

       #print(subdata)

       print(osp.join(processed_dir,str(i),
                                 raw_path.split('/')[-3],
                                    raw_path.split('/')[-1].split('.')[0] + '.pt'))



       #print('before', data.x.shape, 'after:', subdata.x.shape)
       #torch.save(subdata, osp.join(processed_dir,str(i),
       #                          raw_path.split('/')[-3],
       #                             raw_path.split('/')[-1].split('.')[0] + '.pt'))

#def _sampling( num_sample, ratio, distance = None):
#    num_subsample = int(num_sample * ratio)
#    if sample_method == 'farthest':
#        indice = sampler(distance, num_subsample)
#    elif sample_method == 'fuse':
#        # 70% farthest, 30% random
#        far_num =int( 0.7 * num_subsample)
#        rand_num = num_subsample - far_num
#        far_indice = sampler(distance, far_num)
#        remain_item = filter_sampled_indice(far_indice, num_sample)
#        rand_indice = np.asarray(random.sample(remain_item, rand_num))
#        indice = np.concatenate((far_indice, rand_indice),0)
#    #elif sample_method = 'avg':
#
#    else:
#        # random
#        indice = np.random.choice(num_subsample, num_sample, replace=False)
#    return  indice, num_subsample

if __name__ == '__main__':
    # important: to select sample method, change sample_method in both main function and def gen
    setting = CrossValidSetting()
    folds = ['fold_1', 'fold_2', 'fold_3']
    original_files = []
    sampling_ratio = 0.5
    #sample_method = 'fuse'
    sample_method = 'avg'
    mask = 'cia'
    graph_sampler = 'knn'
    epoch = 1
    #sampler = FarthestSampler()
    print(setting.root)

    #for fold in folds:
    #    for f in glob.iglob(setting.root + '/proto/feature/CRC/' + fold + '/**/*', recursive=True):
    #        if '.npy' in f:
    #            original_files.append(f.strip('.npy'))
    processed_dir = os.path.join(setting.root, 'proto',
                                      'fix_%s_%s_%s' % (sample_method, mask, graph_sampler))

    print(processed_dir)
    for f in folds:

        for j in range(epoch):
            print(epoch, j)
            # print('path is : ' + str(osp.join(processed_dir, '%d' % j, f)))
            mkdirs(osp.join(processed_dir, '%d' % j, f))

    # print("original_files : " + str(original_files))
    lmdb_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_lmdb/extended_crc/feat/'
    lmdb_folder = LMDBFolder('/data/smb/syh/PycharmProjects/CGC-Net/data_lmdb/extended_crc/feat/')
    lmdb_dataset = LMDBDataset(lmdb_folder, 1792)
    file_path_lists = lmdb_dataset.file_path_lists

    #res = []
    #for file_fp in file_path_lists:
    #    full_path = Path(str(file_fp))
    #    print(full_path)
    #    res.append(full_path.relative_to(lmdb_path))


    gen = partial(gen, dataset=lmdb_dataset)


    #p = Pool(6)
    print(len(file_path_lists))
    for file_path in file_path_lists:
        gen(file_path)
    #arr = p.map(gen, file_path_lists)
    #p.close()
    #p.join()
    # gen('/data/hdd1/syh/PycharmProjects/CGC-Net/data/proto/feature/CRC/fold_1/1_normal/Patient_013_03_Normal')