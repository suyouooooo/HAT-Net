import os
import glob
import numpy as np

import torch
import torch_geometric




#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_res50/proto/fix_fuse_cia_knn'
#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_res50/proto/fix_add_cia_knn'
path = '/home/baiyu/Extended_CRC_Graph/proto/fix_avg_cia_knn/'

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
#count_1 = 0
#count_2 = 0
#count_3 = 0
#sum1 = 0
#sum2 = 0
#sum3 = 0
def compute_mean(pt_files):
    count = 0
    sums = 0
    for pt_file in pt_files:
        data = torch.load(pt_file)
        #x = data['feat']
        count += data.x.shape[0]
        sums += torch.sum(data.x, dim=0)

    return sums / count


#for epoch in os.listdir(path):
#    for s_idx, s in enumerate(sub_folder):
#        for i in  glob.iglob(os.path.join(path, epoch, s, '*')):
#            data = torch.load(i)
#
#            count[s_idx] += data.x.shape[0]
#            sums[s_idx] += torch.sum(data.x, dim=0)
#
#
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

def compute_std(pt_fps, mean):
    sums = 0
    count = 0
    for fp in pt_fps:
        data = torch.load(fp)
        diff = data.x - mean
        square = torch.square(diff)
        sums += torch.sum(square, dim=0)
        count += data.x.shape[0]

    return torch.sqrt(sums / count)

path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph_Aug/0/proto/fix_full_cia_knn'
import glob


fps = glob.glob(os.path.join(path, '**', '*.pt'), recursive=True)
mean = compute_mean(fps)
std = compute_std(fps, mean)

print(mean)
print(std)

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
