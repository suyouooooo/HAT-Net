import os
import glob

import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import degree


class BACH(Dataset):
    def __init__(self, root, transforms=None):
        super().__init__(root)
        self.root = root
        search_path = os.path.join(self.root, '**', '*.pt')
        self.file_names = []
        self.idxlist = []
        for fp in glob.iglob(search_path, recursive=True):
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

        self.num_nodes = 1070


    def filepath(self):
        return self.file_names

    def len(self):
        return len(self.file_names)

    def get(self, idx):
        fp = self.file_names[idx]
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
        # avoid 0 division
        x = x / diff
        # avoid 0 division
        mean = mean / diff

        # avoid 0 division
        std = std / diff


        # normalize to 0 mean, 1 std
        x = (x - mean) / (std + self.eps) # avoid 0 division

        return x

    #def normalize(self, x):
    #    #[10, -3]   # max 11, min -5
    #    diff = self.max - self.min
    #    x = x - self.min  # larger than 0
    #    x = x / diff  # normalize to 0 - 1
    #    #x += self.max
    #    mean = (self.mean - self.min) / diff # normlized mean
    #    std = (self.std - self.min) / diff  # normalized std

    #    return (x - mean) / std
