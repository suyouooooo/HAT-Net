import os
import random
import glob
import time
import argparse

import numpy as np

import torch
import torch_geometric
from torch.nn.functional import adaptive_avg_pool1d



def pooling(x):
    x = x.unsqueeze(0)
    x = adaptive_avg_pool1d(x, (256))
    x = x.squeeze()
    return x


def compute_mean(pathes):
    count = 0
    sums = 0
    for path in pathes:
        data = torch.load(path)
        #data.x = pooling(data.x)
        #data.x = data.x.unsqueeze(0)
        #data.x = adaptive_avg_pool1d(data.x, (256))
        #data.x = data.x.squeeze()
        count += data.x.shape[0]
        #sums += torch.sum(data.x, dim=0)

        #data = torch.cat([data.x, data.pos], dim=1)
        sums += torch.sum(data.x, dim=0)

    return sums / count

def max_min(pathes):
    max_value = None
    min_value = None
    for path in pathes:
        data = torch.load(path)
        #data.x = pooling(data.x)

        if max_value is None:
            max_value = data.x.max(dim=0)[0]
            continue
        if min_value is None:
            min_value = data.x.min(dim=0)[0]
            continue

        #print(max_value.shape, data.x.shape)
        min_value = torch.cat([min_value.unsqueeze(0), data.x], dim=0).min(dim=0)[0]
        max_value = torch.cat([max_value.unsqueeze(0), data.x], dim=0).max(dim=0)[0]
        #print(min_value.shape)
        #print(max_value.shape)
        #count += data.x.shape[0]
        ##data = torch.cat([data.x, data.pos], dim=1)
        #diff = data.x - mean
        #square = torch.square(diff)
        #sums += torch.sum(square, dim=0)

    #return sums / coun
    return min_value, max_value

def compute_std(pathes, mean=None):
    if mean is None:
        mean = compute_mean(pathes)

    count = 0
    sums = 0
    for path in pathes:
        data = torch.load(path)
        #data.x = pooling(data.x)

        count += data.x.shape[0]
        #data = torch.cat([data.x, data.pos], dim=1)
        diff = data.x - mean
        square = torch.square(diff)
        sums += torch.sum(square, dim=0)

    return sums / count

def dataset_stats(pathes):
    count = 0

    max_value = None
    min_value = None

    feat_sum = 0
    feat_sum_square = 0

    start = time.time()
    for idx, path in enumerate(pathes):
        data = torch.load(path)
        #data.x = pooling(data.x)

        if max_value is None:
            max_value = data.x.max(dim=0)[0]
        if min_value is None:
            min_value = data.x.min(dim=0)[0]

        #print(max_value.shape, data.x.shape)
        min_value = torch.cat([min_value.unsqueeze(0), data.x], dim=0).min(dim=0)[0]
        max_value = torch.cat([max_value.unsqueeze(0), data.x], dim=0).max(dim=0)[0]

        count += data.x.shape[0]
        feat_sum += torch.sum(data.x, dim=0)
        feat_sum_square += torch.sum(torch.square(data.x), dim=0)

        if idx % 100 == 0:
            finish = time.time()
            print('[{}/{}], avg spped: {:02f}'.format(idx + 1, len(pathes), (idx + 1) / (finish - start)))

       # if torch.isnan(feat_sum).sum() != 0:
       #     print('sum')
       #     print(torch.isnan(feat_sum))
       # if torch.isnan(feat_sum_square).sum() != 0:
       #     print('square')
       #     print(torch.isnan(feat_sum_square).sum())


    mean = feat_sum / count
    num_nodes = torch.tensor(int(count / len(pathes)))
    print('count:', count)
    print('total:', len(pathes))
    print('avg:', num_nodes)
    if torch.isnan(mean).sum() != 0:
        print('mean')
        torch.save(mean, 'mean.pt')

    var = torch.abs(feat_sum_square / count - torch.square(mean))
    std = torch.sqrt(var)
    if torch.isnan(std).sum() != 0:
        print('std', count)
        torch.save(feat_sum_square, 'f_square.pt')
        torch.save(mean, 'mean.pt')
        torch.save(std, 'std.pt')


    return min_value, max_value, mean, std, num_nodes


#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_res50/proto/fix_fuse_cia_knn'
#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/VGGUet'
#path = '/data/hdd1/by/cpc_cell_graph/CPC/fold_2'
#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Res50_withtype_Aug/'
#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Aug_CPC'
#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Aug'
#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_res50/proto/fix_add_cia_knn'
#path = '/home/baiyu/Extended_CRC_Graph/proto/fix_avg_cia_knn/'
#path = '/data/hdd1/by/TCGA_Prostate/Feat_Test/0'
#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph_Aug/0/proto/fix_full_cia_knn/'
#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops'
#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_CPC/'
#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/ImageNetPretrain/'
def main(args):
    #data_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Aug'
    #data_path = 'acc872_prostate_5cropsAug/'
    #data_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/Aug/hatnet2048dim/'
    #data_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/Aug/hatnet2048dim/'
    #data_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/Aug/hatnet512dim/'
    #data_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Aug_CGC16dim/'
    #data_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/VGGUet_438dim/'
    #data_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/PanNukeEx6classes/proto/fix_avg_cia_knn/'
    #data_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_res50/proto/fix_add_cia_knn/'
    #data_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_res50/proto/fix_avg_cia_knn/'
    #data_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Cell_Graph/1792_Avg_64/proto/fix_avg_cia_knn/'
    #data_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_res50/proto/fix_avg_cia_knn_128x128/'
    #data_path = 'fix_avg_cia_knn_256x256'
    #data_path = 'fix_avg_cia_knn_512x512/'
    #data_path = 'fix_avg_cia_knn_1024x1024/'
    #data_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_res50/proto/fix_fuse_cia_knn/0/'
    data_path = args.path
    #data_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/Aug/hatnet128dim/'
    #data_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/Aug/hatnet32dim/'
    #data_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/Aug/cgc16dim/'
    #save_path = '/data/hdd1/by/HGIN/ttt_del1'
    #save_path = 'acc872_prostate_5cropsAug/stats'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/test/hatnet512dim/'
    save_path = data_path
    print('...............')
    pathes = glob.glob(os.path.join(data_path, '**', '*.pt'), recursive=True)
    print(len(pathes))
    #pathes.extend(glob.glob(os.path.join('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/test/hatnet512dim', '**', '*.pt'), recursive=True))
    #pathes.extend(glob.glob(os.path.join('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Aug_CGC16dim/', '**', '*.pt'), recursive=True))
    #pathes.extend(glob.glob(os.path.join('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/test/cgc16dim/', '**', '*.pt'), recursive=True))
    print(len(pathes))

    #minmax = max_min(pathes)
    #print(minmax)
    min_value, max_value, mean, std, num_nodes = dataset_stats(pathes)
    print(min_value)
    print()
    print(max_value)
    #np.save(os.path.join('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Aug', 'minmax.npy'), minmax)
    #np.save(os.path.join(save_path, 'minmax.npy'), minmax)
    np.save(os.path.join(save_path, 'min.npy'), min_value)
    np.save(os.path.join(save_path, 'max.npy'), max_value)
    np.save(os.path.join(save_path, 'num_nodes.npy'), num_nodes)


    #print('before:', len(pathes))
    #pathes = random.sample(pathes, k=int(len(pathes) / 5))
    #print('after:', len(pathes))

    #mean = compute_mean(pathes)
    print()
    print(mean)
    #path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Aug'
    #np.save('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_CPC/mean.npy', mean)
    np.save(os.path.join(save_path, 'mean.npy'), mean)


    #std = compute_std(pathes, mean)
    print()
    print(std)
    #np.save('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_CPC/std.npy', std)
    #np.save('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_CPC/std.npy', std)
    np.save(os.path.join(save_path, 'std.npy'), std)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ## yihan
    ##
    parser.add_argument('--path', required=True, type=str)
    args = parser.parse_args()
    main(args)
#sub_folder = ['fold_1', 'fold_2', 'fold_3']
#mean = {
#    1:np.array([0] * 18),
#    2:np.array([0] * 18),
#    3:np.array([0] * 18),
#}
#count = [0, 0, 0]
#sums = {
#    0:torch.tensor([0.0] * 18),
#    1:torch.tensor([0.0] * 18),
#    2:torch.tensor([0.0] * 18)
#}
##count_1 = 0
##count_2 = 0
##count_3 = 0
##sum1 = 0
##sum2 = 0
##sum3 = 0
#for epoch in os.listdir(path):
#    for s_idx, s in enumerate(sub_folder):
#        for i in  glob.iglob(os.path.join(path, epoch, s, '*')):
#            data = torch.load(i)
#
#            count[s_idx] += data.x.shape[0]
#            sums[s_idx] += torch.sum(data.x, dim=0)
##
##
#mean[0] = sums[0] / count[0]
#mean[1] = sums[1] / count[1]
#mean[2] = sums[2] / count[2]
#
#print(sums, count)
#print('mean')
#print(sums[0] / count[0], sums[1] / count[1], sums[2] / count[2])
#
#std = {
#    0:np.array([0] * 18),
#    1:np.array([0] * 18),
#    2:np.array([0] * 18),
#}
#
#count = [0, 0, 0]
#sums = {
#    0:torch.tensor([0.0] * 18),
#    1:torch.tensor([0.0] * 18),
#    2:torch.tensor([0.0] * 18)
#}
#
#for epoch in os.listdir(path):
#    for s_idx, s in enumerate(sub_folder):
#        for i in  glob.iglob(os.path.join(path, epoch, s, '*')):
#            data = torch.load(i)
#
#            count[s_idx] += data.x.shape[0]
#            diff = data.x - mean[s_idx]
#            square = torch.square(diff)
#            sums[s_idx] += torch.sum(square, dim=0)
#
#
#std[0] = sums[0] / count[0]
#std[1] = sums[1] / count[1]
#std[2] = sums[2] / count[2]
#
#std[0] = torch.sqrt(std[0])
#std[1] = torch.sqrt(std[1])
#std[2] = torch.sqrt(std[2])
#
#print('--------------')
#print('std')
#print(std)

#a = torch.tensor([ 1.5406e+11,  1.6204e+10,  4.4896e+11, -1.0955e+08,  4.5582e+09,
#         7.5606e+09,  1.5839e+08,  1.2719e+08,  2.1141e+07,  7.8171e+08,
#         1.4536e+11,  1.8928e+10,  9.8136e+09,  4.7480e+10,  9.3448e+08,
#         8.7070e+10,  1.8978e+12,  1.9000e+12])
