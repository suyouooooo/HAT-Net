
import os.path as osp
import os
from pathlib import Path
import pickle
import string
import time

import sys
from pathlib import Path
from functools import partial
sys.path.append(os.getcwd())
import torch
#from multiprocessing import Pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import radius_graph
from torch_geometric.nn.pool import fps

import copy
import random
import numpy as np
import cv2
import glob
import lmdb
#from common.utils import mkdirs,FarthestSampler,filter_sampled_indice, mkdir

#from setting import CrossValidSetting
#from dataflow.graph_sampler import random_sample_graph2


from torch_cluster import grid_cluster
from torch_scatter import scatter
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos

from stich import LMDBFolder, LMDBDataset, PtFolders, PtDataset

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

#def read_data(dataset, raw_fp):
#    res = dataset.get_file_by_path(raw_fp)
#    feats = res['feat']
#    coords = res['coord']
#
#    feats = np.concatenate((feats, coords), axis= -1)
#    coords = torch.from_numpy(coords).to(torch.float)
#    feats = torch.from_numpy(feats).to(torch.float)
#
#    if '1_normal' in  raw_fp:
#        label = 0
#    elif '2_low_grade' in raw_fp:
#        label = 1
#    elif '3_high_grade' in raw_fp:
#        label = 2
#    else:
#        raise ValueError('value error')
#
#    y = torch.tensor([label], dtype=torch.long)
#    data = Data(x=feats, pos=coords, y=y)
#
#    return data

#def gen(raw_path, dataset, max_neighbours, epoch, sample_method):
#    # important: to select sample method, change sample_method in both main function and def gen
#    #max_neighbours = 8
#    #epoch = 1
#    #graph_sampler = 'knn'
#    #mask = 'cia'
#    ##sample_method= 'fuse'
#    #sample_method= 'avg'
#    setting = CrossValidSetting()
#    processed_dir = os.path.join(setting.root, 'proto',
#                                 'fix_%s_%s_%s' % (sample_method, 'cia', 'knn'))
#
#     # Read data from `raw_path`
#     ## raw_path is /data/hdd1/syh/PycharmProjects/CGC-Net/data/raw/CRC/fold_1/1_normal/xx.png
#     # raw_path is /data/hdd1/syh/PycharmProjects/CGC-Net/data/proto/feature/CRC/fold_1/1_normal/xx(无后缀)
#    data = read_data(dataset, raw_path)
#     # sample epoch time
#    num_nodes = data.x.shape[0]
#    if num_nodes == 0:
#        return
#    #num_sample = num_nodes
#    for i in range(epoch):
#       subdata = copy.deepcopy(data)
#       clusters = grid_cluster(subdata.pos, torch.Tensor([64, 64]))
#       subdata = avg_pool(clusters, subdata)
#
#       # generate the graph
#       #if graph_sampler == 'knn':
#       edge_index = radius_graph(subdata.pos, 100, None, True, max_neighbours)
#       #else:
#       #    edge_index = random_sample_graph2(choice, distance, 100, True,
#                                     #n_sample=8,sparse=True)
#       subdata.edge_index=edge_index
#
#       raw_path1 = Path(raw_path)
#       parts = raw_path1.parts
#       save_fp = os.path.join(processed_dir, str(i), parts[0], parts[2].replace('.npy', '.pt'))
#       print(save_fp)
#       #torch.save(subdata, save_fp)
#       #print(osp.join(processed_dir,str(i),
#       #                          raw_path.split('/')[-3],
#       #                             raw_path.split('/')[-1].split('.')[0] + '.pt'))
#
#
#
#       #print('before', data.x.shape, 'after:', subdata.x.shape)
#       #torch.save(subdata, osp.join(processed_dir,str(i),
#       #                          raw_path.split('/')[-3],
#       #                             raw_path.split('/')[-1].split('.')[0] + '.pt'))

def gen_cell_graph(data):
    subdata = copy.deepcopy(data)

    edge_index = radius_graph(subdata.pos, 100, None, True, 8)
       #else:
       #    edge_index = random_sample_graph2(choice, distance, 100, True,
                                     #n_sample=8,sparse=True)
    subdata.edge_index=edge_index
    return subdata

def gen_label(path):
    name = os.path.basename(path)
    label = int(name.split('_grade_')[1][0]) - 1
    return label

def read_data(res, path):
    feats = res['feat']
    coords = res['coord']

    #feats = np.concatenate((feats, coords), axis= -1)
    if isinstance(coords, (np.ndarray, np.generic)) and isinstance(feats, (np.ndarray, np.generic)):
        coords = torch.from_numpy(coords).to(torch.float)
        feats = torch.from_numpy(feats).to(torch.float)

    #if '1_normal' in  raw_fp:
    #    label = 0
    #elif '2_low_grade' in raw_fp:
    #    label = 1
    #elif '3_high_grade' in raw_fp:
    #    label = 2
    #else:
    #    raise ValueError('value error')
    label = gen_label(path)

    y = torch.tensor([label], dtype=torch.long)
    data = Data(x=feats, pos=coords, y=y)
    #print(path, y, data.y)
    #print(data)
    #print(data)

    return data

class ExCRCPt:
    def __init__(self, path, image_size):

        folder = PtFolders(path)
        self.dataset = PtDataset(folder, image_size)

        print(len(self.dataset))
        #self.npy_names = []
        #for pt in glob.iglob(os.path.join(path, '**', '*.pt'), recursive=True):
        #    self.npy_names.append(pt)

        #self.env = lmdb.open(path, map_size=1099511627776, readonly=True, lock=False)
        #with self.env.begin(write=False) as txn:
        #    self.image_names= [key.decode() for key in txn.cursor().iternext(keys=True, values=False)]

        #cache_file = '_cache_' + ''.join(c for c in path if c in string.ascii_letters)
        #cache_path = os.path.join(path, cache_file)
        #if os.path.isfile(cache_path):
        #    self.npy_names = pickle.load(open(cache_path, "rb"))
        #else:
        #    with self.env.begin(write=False) as txn:
        #        self.npy_names = [key for key in txn.cursor().iternext(keys=True, values=False)]
        #    pickle.dump(self.npy_names , open(cache_path, "wb"))

        #self.npy_names = [key.decode() for key in self.npy_names]

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.dataset)

    def get(idx):
        return self[idx]

    def __getitem__(self, idx):
        #npy_path = self.npy_names[idx]
        #data = torch.load(npy_path)
        path, data = self.dataset[idx]
        #print('mmmmmmmmmm', path, idx, data)

        #with self.env.begin(write=False) as txn:
        #    data = txn.get(npy_path.encode())
        #    data = pickle.loads(data)

        #print(npy_path, data, idx, '111111111111')
        #for key, value in data.items():
        #    print(key, value.shape, idx, '222222222222')
        data = read_data(data, path)
        #print(data, 'fkfkfkfk')
        #print(data, npy_path, idx, '33333333333333')
        if len(data.x) <= 5:
            #idx = random.choice()
            idx = 1
            path, data = self.dataset[idx]
            data = read_data(data, path)
            #data = gen_cell_graph(data)
            #
            #print(data)
            #return data

        data.path = path
        data = fuse(data)
        data = gen_cell_graph(data)
        #print('gen', data.path, data.y)
        #import sys; sys.exit()
        return data

def fuse(data):
    """data.x, data.pos"""
    length = len(data.x)
    l1 = length
    sample = torch.rand(length).topk(int(length // 2)).indices
    mask = torch.zeros(length, dtype=torch.bool)
    mask.scatter_(dim=0, index=sample, value=True)
    #print(mask.shape, data)
    #print((mask == 1).sum())

    fps_x = data.x[mask]
    fps_pos = data.pos[mask]
    l2 = len(fps_x)
    #print(fps_x.shape)
    #print(fps_pos.shape)
    #fps_data = Data(x=fps_x, pos=fps_pos)

    #fps_x = fps_x[:2]
    batch = torch.zeros((len(fps_pos))).long()

    #x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * 300
    #print(x.dtype)
    #x = fps_pos[:4].float()
    #print(x.dtype)
    #batch = torch.tensor([0, 0, 0, 0])
    #batch = torch.zeros((len(x))).long()
    #print(x.shape, fps_pos.shape)
    #print(x.shape, batch.shape)
    #batch = torch.tensor([0, 0, 0, 0])
    #fps_idx = fps(x, ratio=0.5)
    #fps_idx = fps(x, ratio=0.5)
    #if len(fps_pos) == 0:
        #print(data, fps_pos)
    #try:
    fps_idx = fps(fps_pos.float(), batch=batch, ratio=0.7, random_start=False)
    #except Exception as e:
        #print(e, fps_pos)


    l3 = len(fps_idx)
    #print(fps_idx, 111111111111111)
    fps_x = fps_x[fps_idx]
    fps_pos = fps_pos[fps_idx]
    fps_data =Data(x=fps_x, pos=fps_pos)

    random_x = data.x[~mask]
    random_pos = data.pos[~mask]

    length = len(random_x)
    sample = torch.rand(length).topk(int(length * 0.3)).indices
    mask = torch.zeros(length, dtype=torch.bool)
    mask.scatter_(dim=0, index=sample, value=True)

    random_x = random_x[mask]
    random_pos = random_pos[mask]
    l4 = len(random_x)
    #random_data = Data(x=random_x, pos=fps_pos)

    fps_data.x = torch.cat([fps_data.x, random_x], dim=0)
    fps_data.pos = torch.cat([fps_data.pos, random_pos], dim=0)

    data.x = fps_data.x
    data.pos = fps_data.pos

    return data


class ProsatePt:
    def __init__(self, path):

        self.npy_names = []
        for pt in glob.iglob(os.path.join(path, '**', '*.pt'), recursive=True):
            self.npy_names.append(pt)

        #self.env = lmdb.open(path, map_size=1099511627776, readonly=True, lock=False)
        #with self.env.begin(write=False) as txn:
        #    self.image_names= [key.decode() for key in txn.cursor().iternext(keys=True, values=False)]

        #cache_file = '_cache_' + ''.join(c for c in path if c in string.ascii_letters)
        #cache_path = os.path.join(path, cache_file)
        #if os.path.isfile(cache_path):
        #    self.npy_names = pickle.load(open(cache_path, "rb"))
        #else:
        #    with self.env.begin(write=False) as txn:
        #        self.npy_names = [key for key in txn.cursor().iternext(keys=True, values=False)]
        #    pickle.dump(self.npy_names , open(cache_path, "wb"))

        #self.npy_names = [key.decode() for key in self.npy_names]

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.npy_names)

    def get(idx):
        return self[idx]

    def __getitem__(self, idx):
        npy_path = self.npy_names[idx]
        data = torch.load(npy_path)
        #with self.env.begin(write=False) as txn:
        #    data = txn.get(npy_path.encode())
        #    data = pickle.loads(data)

        #print(npy_path, data, idx, '111111111111')
        #for key, value in data.items():
        #    print(key, value.shape, idx, '222222222222')
        #shape = 'before', data['feat'].shape
        data = read_data(data, npy_path)
        #aa = len(data.x)
        #data = fuse(data)
        #print(aa, data.x.shape)
        #print(data, npy_path, idx, '33333333333333')
        data = gen_cell_graph(data)
        data.path = npy_path
        #print(shape, 'after', data)
        #print('gen', data.path, data.y)
        return data

class Prosate:
    def __init__(self, path):

        self.env = lmdb.open(path, map_size=1099511627776, readonly=True, lock=False)
        #with self.env.begin(write=False) as txn:
        #    self.image_names= [key.decode() for key in txn.cursor().iternext(keys=True, values=False)]

        cache_file = '_cache_' + ''.join(c for c in path if c in string.ascii_letters)
        cache_path = os.path.join(path, cache_file)
        if os.path.isfile(cache_path):
            self.npy_names = pickle.load(open(cache_path, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.npy_names = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.npy_names , open(cache_path, "wb"))

        self.npy_names = [key.decode() for key in self.npy_names]

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.npy_names)

    def get(idx):
        return self[idx]

    def __getitem__(self, idx):
        npy_path = self.npy_names[idx]
        with self.env.begin(write=False) as txn:
            data = txn.get(npy_path.encode())
            data = pickle.loads(data)

        data = read_data(data, npy_path)
        data = gen_cell_graph(data)
        data.path = npy_path
        #print('gen', data.path, data.y)
        return data


def visualize(image, coord):
    for cen in coord:
        cen = [int(c) for c in cen]
        image = cv2.circle(image, tuple(cen)[::-1], 3, (0, 200, 0), cv2.FILLED, 1)

    return image

def write_data(save_path, data):
    #print(save_path)
    save_path = os.path.join(save_path, os.path.basename(data.path.replace('.jpg', '.pt')))
    #print(111,  save_path)
    #print(data)
    #sys.exit()
    if len(data.x) <= 5:
        return

    #print(save_path)
    torch.save(data.clone(), save_path)

def add_sub_folder(save_path):
    return os.path.join(save_path, 'proto', 'fix_full_cia_knn')

def get_lmdb_pathes(lmdb_path):
    res = []
    for i in glob.iglob(os.path.join(lmdb_path, '**', 'data.mdb')):
        res.append(os.path.dirname(i))
    return res

def gen_save_folder(lmdb_fp):
    save_path = lmdb_fp.replace('Feat_Test', 'Cell_Graph_Test')
    #print(save_path)
    #import sys;sys.exit()
    #save_path = add_sub_folder(save_path)
    return save_path

def pt_folders(root):
    pathes = set()
    for i in glob.iglob(os.path.join(root, '**', '*.pt'), recursive=True):
        prefix = os.path.dirname(i)
        pathes.add(prefix)
    return list(pathes)

def write_batch(save_path, batch):
    for i in range(batch.num_graphs):
        write_data(save_path, batch[i])

class DataSetFactory:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, path, image_size=224 * 8):
        #return self.dataset(path, image_size)
        return self.dataset(path)

def run(pathes, data_fact, save_path):


    start = time.time()
    for fp in pathes:
        #print(fp)
        #data_set = ProsatePt(fp)
        #data_set = ExCRCPt(fp, 224 * 8)
        data_set = data_fact(fp)
        data_loader = DataLoader(data_set, num_workers=4, batch_size=32, shuffle=True)
        #save_path = gen_save_folder(fp)
        #save_path
        #print(2222, save_path)
        os.makedirs(save_path, exist_ok=True)
        #import sys; sys.exit()
        for idx, b in enumerate(data_loader):
            #print(b)
            #print(33333, save_path)
            #import sys; sys.exit()
            write_batch(save_path, b)
            finish = time.time()
            avg_speed = (idx + 1) * len(b) / (finish - start)
            print('[{}/{}]....avg speed:{:2f}, {}'.format(idx + 1, len(data_loader), avg_speed, save_path))

def gen_ecrc_save(save_path):
    subdirs = [
        'fold_1/1_normal/',
        'fold_1/2_low_grade/',
        'fold_1/3_high_grade/',

        'fold_2/1_normal/',
        'fold_2/2_low_grade/',
        'fold_2/3_high_grade/',

        'fold_3/1_normal/',
        'fold_3/2_low_grade/',
        'fold_3/3_high_grade/',
    ]
    return [os.path.join(save_path, p) for p in subdirs]

def gen_ecrc_root(root_path):
    subdirs = [
        'fold_1/1_normal/',
        'fold_1/2_low_grade/',
        'fold_1/3_high_grade/',

        'fold_2/1_normal/',
        'fold_2/2_low_grade/',
        'fold_2/3_high_grade/',

        'fold_3/1_normal/',
        'fold_3/2_low_grade/',
        'fold_3/3_high_grade/',
    ]
    root = [os.path.join(root_path, p) for p in subdirs]
    return root

def gen_prostate_root(root_path):
    return [root_path]

def gen_prostate_save(save_path):
    return [save_path]

if __name__ == '__main__':
    #image_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/images/'
    #lmdb_root = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat_Aug/'
    #lmdb_pathes = get_lmdb_pathes(lmdb_root)
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/'

    #root = '/data/hdd1/by/TCGA_Prostate/Feat_Test_No_Random/'
    #root = '/data/hdd1/by/TCGA_Prostate/Feat_Aug/0'
    #save_path = '/data/hdd1/by/TCGA_Prostate/Cell_Graph_Test_No_Random/0'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph_Aug/0/proto/fix_full_cia_knn/'
    #root = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat_Aug/Before_FC/0'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph_Aug/Before_FC'
    #root = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat/5Crops_Aug'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Aug'
    #root = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat_Aug/test'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/test'
    #root = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat_Aug/train'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/train'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Cell_Graph/1792_Fuse_64_dim16/fold_3/1_normal'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Cell_Graph/1792_Fuse_64_dim16/fold_2/1_normal'

    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Cell_Graph/1792_Fuse_64_dim16/fold_1/2_low_grade/'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Cell_Graph/1792_Fuse_64_dim16/fold_1/3_high_grade/'

    #root = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat_hand_crafted/0/'
    #root = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat_hand_crafted/0/fold_3/1_normal/'
    #root = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat_hand_crafted/0/fold_2/1_normal/'

    #root = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat_hand_crafted/0/fold_1/2_low_grade/'
    #root = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat_hand_crafted/0/fold_1/3_high_grade/'

    #src_prefix = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Feat/CPC/0'
    #save_prefix = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/CPC/'
    #src_prefix = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Feat/VGGUet/0'
    #save_prefix = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/VGGUet'
    #subdirs = [
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat/5Crops_CPC/0/'
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat/5Crops_Aug_CPC/'
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat/ImageNetPretrain/0/fold_1/1_normal/',
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat/ImageNetPretrain/0/fold_1/2_low_grade/',
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat/ImageNetPretrain/0/fold_1/3_high_grade/',
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat/ImageNetPretrain/0/fold_1/1_normal/',
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat/ImageNetPretrain/0/fold_1/2_low_grade/',
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat/ImageNetPretrain/0/fold_1/3_high_grade/',
    #    #'fold_1/1_normal/',
    #    #'fold_1/2_low_grade/',
    #    #'fold_1/3_high_grade/',

    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat/ImageNetPretrain/0/fold_2/1_normal/',
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat/ImageNetPretrain/0/fold_2/2_low_grade/',
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat/ImageNetPretrain/0/fold_2/3_high_grade/',

    #    'fold_2/1_normal/',
    #    'fold_2/2_low_grade/',
    #    'fold_2/3_high_grade/',

    #    #'fold_3/1_normal/',
    #    #'fold_3/2_low_grade/',
    #    #'fold_3/3_high_grade/',
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat/ImageNetPretrain/0/fold_3/1_normal/',
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat/ImageNetPretrain/0/fold_3/2_low_grade/',
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat/ImageNetPretrain/0/fold_3/3_high_grade/',
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat_hand_crafted/0/fold_2/1_normal/',

    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat_hand_crafted/0/fold_1/2_low_grade/',
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat_hand_crafted/0/fold_2/2_low_grade/',
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat_hand_crafted/0/fold_3/2_low_grade/',

    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat_hand_crafted/0/fold_1/3_high_grade',
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat_hand_crafted/0/fold_2/3_high_grade',
    #    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat_hand_crafted/0/fold_3/3_high_grade',
    #]
    ##save_path = [
    ##    '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/ImageNetPretrain/fold_1/1_normal/',
    ##    '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/ImageNetPretrain/fold_1/2_low_grade/',
    ##    '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/ImageNetPretrain/fold_1/3_high_grade/',

    ##    '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/ImageNetPretrain/fold_2/1_normal/',
    ##    '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/ImageNetPretrain/fold_2/2_low_grade/',
    ##    '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/ImageNetPretrain/fold_2/3_high_grade/',

    ##    '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/ImageNetPretrain/fold_3/1_normal/',
    ##    '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/ImageNetPretrain/fold_3/2_low_grade/',
    ##    '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/ImageNetPretrain/fold_3/3_high_grade/',
    ##    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_CPC'
    ##    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Aug_CPC/'
    ##    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Cell_Graph/1792_Fuse_64_dim16/fold_2/1_normal',

    ##    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Cell_Graph/1792_Fuse_64_dim16/fold_1/2_low_grade',
    ##    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Cell_Graph/1792_Fuse_64_dim16/fold_2/2_low_grade',
    ##    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Cell_Graph/1792_Fuse_64_dim16/fold_3/2_low_grade',

    ##    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Cell_Graph/1792_Fuse_64_dim16/fold_1/3_high_grade',
    ##    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Cell_Graph/1792_Fuse_64_dim16/fold_2/3_high_grade',
    ##    #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Cell_Graph/1792_Fuse_64_dim16/fold_3/3_high_grade',
    ##]
    #root = [os.path.join(src_prefix, p) for p in subdirs]
    #save_path = [os.path.join(save_prefix, p) for p in subdirs]

    root = gen_prostate_root('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat/5Crops_Res50_withtype_Aug')
    save_path = gen_prostate_save('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Res50_withtype_Aug')

    for r, s in zip(root, save_path):
        print(r, s)
        pathes = pt_folders(r)
        #print(pathes)
        #os.system('ls {} | wc -l'.format(r))
        #continue
        #data_fact = DataSetFactory(ExCRCPt)
        data_fact = DataSetFactory(ProsatePt)

        run(pathes, data_fact, s)
    #for fp in pathes:
    #    #print(fp)
    #    #data_set = ProsatePt(fp)
    #    #data_set = ExCRCPt(fp, 224 * 8)
    #    data_set = data_fact(fp)
    #    data_loader = DataLoader(data_set, num_workers=2, batch_size=16, shuffle=True)
    #    #save_path = gen_save_folder(fp)
    #    #save_path
    #    #print(2222, save_path)
    #    os.makedirs(save_path, exist_ok=True)
    #    #import sys; sys.exit()
    #    for idx, b in enumerate(data_loader):
    #        #print(b)
    #        #print(33333, save_path)
    #        #import sys; sys.exit()
    #        write_batch(save_path, b)
    #        print('[{}/{}].... {}'.format(idx + 1, len(data_loader), save_path))
