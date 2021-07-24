
import os.path as osp
import os

import sys
import os
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
    data.pos = pos

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



FEATURE_NAMES = ['max_gray','min_gray','mean_im_out','diff','mean_h','mean_s','mean_v','mean_l','mean_a','mean_b','var_im'
                ,'skew_im','mean_ent','glcm_dissimilarity','glcm_homogeneity','glcm_energy','glcm_ASM','hull_area',
                'eccentricity','equiv_diameter','extent','area','majoraxis_length','minoraxis_length','perimeter',
                'solidity','orientation','radius','aspect_ratio']

_CROSS_VAL = {1:{'train':['fold_1', 'fold_2'], 'valid': ['fold_3']},
              2:{'train':['fold_1', 'fold_3'], 'valid': ['fold_2']},
              3:{'train':['fold_2', 'fold_3'], 'valid': ['fold_1']},

}

_MEAN_CIA = {1:[ 1.44855589e+02,  1.50849152e+01,  4.16993829e+02, -9.89115031e-02,
         4.29073361e+00,  7.03308534e+00,  1.50311764e-01,  1.20372119e-01,
         1.99874447e-02,  7.24825770e-01,  1.28062193e+02,  1.71914904e+01,
         9.00313323e+00,  4.29522533e+01,  8.76540101e-01,  8.06801284e+01, 3584,3584],
         2:[ 1.45949547e+02,  1.53704952e+01,  4.39127922e+02, -1.10080479e-01,
         4.30617772e+00,  7.27624697e+00,  1.45825849e-01,  1.21214980e-01,
         2.03645262e-02,  7.28225987e-01,  1.27914898e+02,  1.72524907e+01,
         8.96012595e+00,  4.30067152e+01,  8.76016742e-01,  8.09466370e+01,3584,3584],
         3:[ 1.45649518e+02,  1.52438912e+01,  4.30302592e+02, -1.07054163e-01,
         4.29877990e+00,  7.13800092e+00,  1.47971754e-01,  1.20517868e-01,
         2.00830612e-02,  7.24701226e-01,  1.26430193e+02,  1.71710396e+01,
         8.94070628e+00,  4.27421136e+01,  8.74665450e-01,  8.02611304e+01,3584,3584]}

_STD_CIA = {1:[3.83891570e+01, 1.23159786e+01, 3.74384781e+02, 5.05079918e-01,
        1.91811771e-01, 2.95460595e+00, 7.31040425e-02, 7.41484835e-02,
        2.84762625e-02, 2.47544275e-01, 1.51846534e+02, 5.96200235e+01,
        6.00087195e+00, 2.85961395e+01, 1.95532620e-01, 5.49411936e+01,3584,3584],
            2:[3.86514982e+01, 1.25207234e+01, 3.87362858e+02, 5.02515226e-01,
        1.89045551e-01, 3.05856764e+00, 7.22404102e-02, 7.53090608e-02,
        2.90460236e-02, 2.46734916e-01, 1.53743958e+02, 6.34661492e+01,
        6.02575043e+00, 2.88403590e+01, 1.94214810e-01, 5.49984596e+01,3584,3584],
            3:[3.72861596e+01, 1.23840868e+01, 3.87834784e+02, 5.02444847e-01,
        1.86722327e-01, 2.99248449e+00, 7.20327363e-02, 7.45553798e-02,
        2.87285660e-02, 2.49195190e-01, 1.50986869e+02, 6.56370060e+01,
        6.00008814e+00, 2.86376250e+01, 1.97764021e-01, 5.54134874e+01,3584,3584]}




def _read_one_raw_graph( raw_file_path):
    # import pdb;pdb.set_trace()
    nodes_features = np.load( raw_file_path + '.npy')
    coordinates = np.load(raw_file_path.replace('feature', 'coordinate') + '.npy')
    nodes_features = np.concatenate((nodes_features, coordinates), axis= -1)
    coordinates = torch.from_numpy(coordinates).to(torch.float)
    nodes_features = torch.from_numpy(nodes_features ).to(torch.float)
    if '1_normal' in raw_file_path:
        label = 0
    elif '2_low_grade' in raw_file_path:
        label = 1
    else:
        label = 2
    y = torch.tensor([label], dtype=torch.long)
    data = Data(x = nodes_features, pos = coordinates, y = y)
    return data


def gen(raw_path):
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
    data =_read_one_raw_graph(raw_path)
     # sample epoch time
    num_nodes = data.x.shape[0]
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

       # generate the graph
       if graph_sampler == 'knn':
           edge_index = radius_graph(subdata.pos, 100, None, True, max_neighbours)
       else:
           edge_index = random_sample_graph2(choice, distance, 100, True,
                                     n_sample=8,sparse=True)
       subdata.edge_index=edge_index

       print(osp.join(processed_dir,str(i),
                                 raw_path.split('/')[-3],
                                    raw_path.split('/')[-1].split('.')[0] + '.pt'))

       #print('here1111')
       #print(1, data, avg_data)
       #print(data.pos[3].long(), avg_data.pos[3].long())
       #print()


       #print('before', data.x.shape, 'after:', subdata.x.shape)
       torch.save(subdata, osp.join(processed_dir,str(i),
                                 raw_path.split('/')[-3],
                                    raw_path.split('/')[-1].split('.')[0] + '.pt'))

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
    sampler = FarthestSampler()
    print(setting.root)

    for fold in folds:
        for f in glob.iglob(setting.root + '/proto/feature/CRC/' + fold + '/**/*', recursive=True):
            if '.npy' in f:
                original_files.append(f.strip('.npy'))
    processed_dir = os.path.join(setting.root, 'proto',
                                      'fix_%s_%s_%s' % (sample_method, mask, graph_sampler))
    #for f in folds:

    #    for j in range(epoch):
    #        print(epoch, j)
    #        # print('path is : ' + str(osp.join(processed_dir, '%d' % j, f)))
    #        mkdirs(osp.join(processed_dir, '%d' % j, f))

    # print("original_files : " + str(original_files))
    #p = Pool(6)
    #arr = p.map(gen,original_files )
    #p.close()
    #p.join()

    # gen('/data/hdd1/syh/PycharmProjects/CGC-Net/data/proto/feature/CRC/fold_1/1_normal/Patient_013_03_Normal')
    count = 0
    print(len(original_files))
    for i in original_files:
        count += 1
        if count != 3:
            continue

        print('ffff')
        print(i + '.npy')
        #nodes_features = np.load(i + '.npy')
        data = _read_one_raw_graph(i)
        print(data)
        clusters = grid_cluster(data.pos, torch.Tensor([64, 64]))
        #print(data.pos)
        #print(clusters[clusters == 1])
        #print(np.unique(clusters))
        #for i in np.unique(clusters):
        print(clusters)
        print(clusters[2295 == clusters])
        mask = clusters == 2295
        print(clusters.max(), 11)
        print(mask)
        print(data.x[mask])
        data = add_pool(clusters, data)
        print(data)
           #print(i)

        clusters = grid_cluster(data.pos, torch.Tensor([64, 64]), start=0, end=torch.Tensor([1792]))
        print(clusters.max())
        #print(torch.ops.torch_cluster)
        #print(clusters == 2295)

        #print(nodes_features[])
        sys.exit()