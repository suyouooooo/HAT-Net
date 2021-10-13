import glob
import os
from pathlib import Path

import torch
#from multiprocessing import Pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import radius_graph
from torch_geometric.nn.pool import fps
import numpy as np


def gen_label(path):
    name = os.path.basename(path)
    label = int(name.split('_grade_')[1][0]) - 1
    return label

def read_data(res, path):
    feats = res['feat']
    coords = res['coord']
    #feats = res[:16]

    #feats = np.concatenate((feats, coords), axis= -1)
    if isinstance(coords, (np.ndarray, np.generic)) and isinstance(feats, (np.ndarray, np.generic)):
        coords = torch.from_numpy(coords).to(torch.float)
        feats = torch.from_numpy(feats).to(torch.float)

    label = gen_label(path)

    y = torch.tensor([label], dtype=torch.long)
    data = Data(x=feats, pos=coords, y=y)

    return data

def gen_cell_graph(data):
    #subdata = copy.deepcopy(data)
    subdata = data.clone()

    edge_index = radius_graph(subdata.pos, 100, None, True, 8)
       #else:
       #    edge_index = random_sample_graph2(choice, distance, 100, True,
                                     #n_sample=8,sparse=True)
    subdata.edge_index=edge_index
    return subdata


class CellGraphPt:
    def __init__(self, path, transforms=None):

        self.npy_names = []
        self.transforms = transforms
        for pt in glob.iglob(os.path.join(path, '**', '*.npy'), recursive=True):
            self.npy_names.append(pt)

        self.path = path

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.npy_names)

    def get(idx):
        return self[idx]

    def __getitem__(self, idx):
        npy_path = self.npy_names[idx]
        rel_image_path = Path(npy_path).relative_to(self.path)
        data = torch.load(npy_path)
        data = read_data(data, npy_path)
        #data = fuse(data)
        #if self.transforms is not None:
        for trans in self.transforms:
            data = trans(data)

        data = gen_cell_graph(data)

        #print(rel_image_path)
        data.path = rel_image_path
        return data


class CellGraphNpy:
    def __init__(self, path, transforms=None):

        self.npy_names = []
        self.transforms = transforms
        for pt in glob.iglob(os.path.join(path, '**', '*.npy'), recursive=True):
            self.npy_names.append(pt)

        self.path = path

        print(len(self))

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.npy_names)

    def get(idx):
        return self[idx]

    def read_data(self, res, path):
        #feats = res['feat']
        #coords = res['coord']
        feats, coords = res
        #feats = res[:16]

        #feats = np.concatenate((feats, coords), axis= -1)
        if isinstance(coords, (np.ndarray, np.generic)) and isinstance(feats, (np.ndarray, np.generic)):
            coords = torch.from_numpy(coords).to(torch.float)
            feats = torch.from_numpy(feats).to(torch.float)

        label = gen_label(path)

        y = torch.tensor([label], dtype=torch.long)
        data = Data(x=feats, pos=coords, y=y)

        return data

    def __getitem__(self, idx):
        npy_path = self.npy_names[idx]
        feat_path = npy_path
        coord_path = npy_path.replace('feature', 'coordinate')
        rel_image_path = Path(npy_path).relative_to(self.path)
        #data = torch.load(npy_path)
        feat = np.load(feat_path)
        coord = np.load(coord_path)
        assert len(feat) == len(coord)
        #data = np.load(npy_path)
        data = self.read_data((feat, coord), npy_path)
        #data = fuse(data)
        #if self.transforms is not None:
        for trans in self.transforms:
            data = trans(data)

        data = gen_cell_graph(data)

        #print(rel_image_path)
        #print(type(rel_image_path))
        data.path = str(rel_image_path).replace('.npy', '.pt')
        return data