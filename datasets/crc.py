import os
import glob

import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import degree

class CRC(Dataset):
    def __init__(self, root, cv, image_set, transforms=None):
        super().__init__(root)
        self.root = root
        search_path = os.path.join(self.root, '**', '*.pt')
        self.file_names = []
        self.idxlist = []
        fold = 'fold_{}'.format(cv)
        for fp in glob.iglob(search_path, recursive=True):
            if image_set == 'train':
                if fold in fp:
                    continue
            if image_set == 'test':
                if fold not in fp:
                    continue

            #if cv == 1:
            #    if 'fold_1' not in fp:
            #        continue
            #elif cv == 2:
            #    if 'fold_2' not in fp:
            #        continue
            #elif cv == 3:
            #    if 'fold_3' not in fp:
            #        continue

            self.file_names.append(fp)
            self.idxlist.append(os.path.basename(fp))
        self.transforms = transforms
        #self.mean = torch.tensor(mean)
        #self.std = torch.tensor(std)
        self.mean = torch.tensor(np.load(
            os.path.join(root, 'mean.npy') # 87.2 withtypes
        ))
        self.std = torch.tensor(np.load(
            #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_CPC/std.npy'
            #'/data/hdd1/by/HGIN/acc872_prostate_5cropsAug/std.npy' # 87.2 withtypes
            os.path.join(root, 'std.npy')
        ))
        #self.min = torch.tensor(np.load(
        #    '/data/hdd1/by/HGIN/acc872_prostate_5cropsAug/stats/min.npy'
        #))
        #self.max = torch.tensor(np.load(
        #    '/data/hdd1/by/HGIN/acc872_prostate_5cropsAug/stats/max.npy'
        #))
        #self.mean = torch.tensor(np.load(
        #    '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Res50_withtype_Aug/mean.npy'
        #))
        #self.std = torch.tensor(np.load(
        #    '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Res50_withtype_Aug/std.npy'
        #))
        self.eps = 1e-7


        self.min = torch.tensor(np.load(
            #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_CPC/std.npy'
            #'/data/hdd1/by/HGIN/acc872_prostate_5cropsAug/std.npy' # 87.2 withtypes
            os.path.join(root, 'min.npy')
        ))
        self.max = torch.tensor(np.load(
            #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_CPC/std.npy'
            #'/data/hdd1/by/HGIN/acc872_prostate_5cropsAug/std.npy' # 87.2 withtypes
            os.path.join(root, 'max.npy')
        ))

        #self.num_nodes = 2271
        self.num_nodes = torch.tensor(np.load(
            #'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_CPC/std.npy'
            #'/data/hdd1/by/HGIN/acc872_prostate_5cropsAug/std.npy' # 87.2 withtypes
            os.path.join(root, 'num_nodes.npy')
        ))


    def filepath(self):
        return self.file_names

    def len(self):
        return len(self.file_names)

    def get(self, idx):
        fp = self.file_names[idx]
        #print(fp)
        data = torch.load(fp)
        #deg = degree(data.edge_index[0])
        #print(deg.max(), 33333333)
        if self.transforms is not None:
            data = self.transforms(data)

        data.patch_idx = torch.tensor([idx])
        #data.x = (data.x - self.mean) / self.std
        #data.x = data.x.float()

        data.x = self.normalize(data.x).float()
        #print(data.x.max(), data.x.min())

        #print(data.x.shape)
        #print(data.x.max(), data.x.min())

        #data.x[torch.isnan(data.x)] = 0
        #data.x[torch.isinf(data.x)] = 0

        return data

    def normalize(self, x):
        #print(x.max(), x.min())
        diff = self.max - self.min + self.eps # avoid 0 division
        x = x - self.min  # larger than 0
        assert (x < 0).sum() == 0
        mean = self.mean - self.min
        std = self.std

        # normalize to range (0, 1)
        #xmax = x.max()
        #xmin = x.min()
        # avoid 0 division
        x = x / diff
        #print(xmax, xmin, x.max(), x.min(), '33333333333333333333333', diff.max(), diff.min(), )
        #print(mean.max(), mean.min())
        #meanmin = mean.min()
        #meanmax = mean.max()
        # avoid 0 division
        mean = mean / diff
        #print(mean.max(), mean.min(), meanmin, meanmax)
        # avoid 0 division
        std = std / diff
        #print(std.max(), std.min(), 'fffffffffffffffffff')
        #print(mean.max(), mean.min(), std.max(), std.min())

        #print(std.max(), std.min(), 111111111)

        # normalize to 0 mean, 1 std
        #tmp = x - mean
        #print(tmp.min(), tmp.max(), 'ccccccccccccccc')
        #tmp[tmp < self.eps] = 0
        #x = tmp / (std + self.eps)
        #x = x - mean
        x = (x - mean) / (std + self.eps) # avoid 0 division
        #print(x.min(), x.max())
        #print((std < self.eps).sum(), id(std), '11')
        #print((x > 10).sum(), id(std))
        #print(x - mean)
        #print(x.max(), x.min())

        return x