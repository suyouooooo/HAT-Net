import json
import os
from pathlib import Path
from functools import partial
import sys
import random
import pickle
import glob
import string
#sys.path.append('../')
sys.path.append(os.getcwd())

import lmdb
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import adaptive_avg_pool3d
from torchvision import transforms
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_cluster import grid_cluster
from torch_scatter import scatter
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos


import cv2
import numpy as np
from skimage.measure import regionprops


from stich import JsonFolder, JsonDataset, ImageFolder, ImageDataset


#class Patches:
#    def __init__(self, image, coords, bboxes, patch_size):
#        """
#            bboxes :bbox [min_y, min_x, max_y, max_x]
#            coords: (row, col)
#        """
#
#        self.image = image
#        self.bboxes = bboxes
#        self.patch_size = patch_size
#        self.coords = coords
#        mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr
#        std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
#        self.transforms = transforms.Compose([
#            transforms.ToPILImage(),
#            transforms.Resize((self.patch_size, self.patch_size)),
#            transforms.ToTensor(),
#            transforms.Normalize(mean, std),
#        ])
#
#
#    def __len__(self):
#        return len(self.bboxes)
#
#    def __getitem__(self, idx):
#        bbox = self.bboxes[idx]
#        #print(bbox, idx)
#        bbox = self.pad_patch(*bbox, self.patch_size)
#        patch = self.image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
#        patch = self.transforms(patch)
#        return patch, np.array(self.coords[idx])
#
#    def pad_patch(self, min_x, min_y, max_x, max_y, output_size):
#        width = max_x - min_x
#        height = max_y - min_y
#
#        if width < output_size:
#            w_diff_left = (output_size - width) // 2
#            w_diff_right = output_size - width - w_diff_left
#            min_x = max(0, min_x - w_diff_left)
#            max_x = max_x + w_diff_right
#        if height < output_size:
#            h_diff_top = (output_size - height) // 2
#            h_diff_bot = output_size - height - h_diff_top
#            min_y = max(0, min_y - h_diff_top)
#            max_y = max_y + h_diff_bot
#
#        return min_x, min_y, max_x, max_y


#class Res50NumpyMaskConfig(Res50BaseConfig):
#    def __init__(self):
#        super().__init__()
#
#        self.image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC/'
#        self.label_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_su/proto/mask/CRC'
#        self.save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_lmdb/crc/res50_1792_avg_knn/proto/'
#        self.pool = 'avg'
#        transform = avg_pooling
#        #self.pool = 'add':
#        #transform = add_pooling
#
#        self.seg = 'cia'
#        folder_name = 'fix_{}_{}_{}'.format(self.pool, self.seg, self.method)
#        self.training_data_path = os.path.join('/home/baiyu/training_data/CRC', folder_name)
#        self.return_mask = True
#        self.dataset = ImageNumpy(self.image_path, self.label_path, self.image_size, self.return_mask)
#        self.writer = LMDBWriter(self.save_path)
#        self.reader = partial(LMDBReader.init_dataset, transform=transform)

class ImageNumpyCrop:
    def __init__(self, image_folder, label_folder, patch_size):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_names = []
        search_path = os.path.join(image_folder, '**', '*.png')
        for image_name in glob.iglob(search_path, recursive=True):
            if 'mask' in image_name:
                continue
            self.image_names.append(image_name)

        self.patch_size = patch_size
        #mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr
        #std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        #self.transforms = transforms.Compose([
        #    transforms.ToPILImage(),
        #    transforms.Resize((self.patch_size, self.patch_size)),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean, std),
        #])

    def __len__(self):
        return len(self.image_names)

    def image2mask_fp(self, image_path):
        rel_image_path = Path(image_path).relative_to(self.image_folder)
        mask_path = os.path.join(self.label_folder, str(rel_image_path).replace('.png', '.npy'))
        return mask_path

    def pad_patch(self, min_x, min_y, max_x, max_y, output_size):
        width = max_x - min_x
        height = max_y - min_y

        if width < output_size:
            w_diff_left = (output_size - width) // 2
            w_diff_right = output_size - width - w_diff_left
            min_x = max(0, min_x - w_diff_left)
            max_x = max_x + w_diff_right
        if height < output_size:
            h_diff_top = (output_size - height) // 2
            h_diff_bot = output_size - height - h_diff_top
            min_y = max(0, min_y - h_diff_top)
            max_y = max_y + h_diff_bot

        return min_x, min_y, max_x, max_y

    def __getitem__(self, idx):
        #import time
        #start = time.time()
        image_path = self.image_names[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        bboxes = []
        coords = []
        patches = []

        #tp1 = time.time()
        mask_path = self.image2mask_fp(image_path)
        #print('time1', tp1 - start)
        mask = np.load(mask_path)
        #tp2 = time.time()
        #print('time2', tp2 - tp1)
        props = regionprops(mask)
        #tp3 = time.time()
        #print('time3', tp3 - tp2)
        #bboxes = []
        for prop in props:
            #bboxes.append([int(c) for c in prop.bbox])

            #patch_before = image[prop.bbox[0]: prop.bbox[2]+1, prop.bbox[1]:prop.bbox[3]+1]
            bbox = self.pad_patch(*prop.bbox, self.patch_size)

            patch = image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
            patch = cv2.resize(image, (self.patch_size, self.patch_size))
            #print(patch.shape, bbox, patch_before.shape, prop.bbox)
            #patch = self.transforms(patch)
            patches.append(patch)
            coords.append([int(c) for c in prop.centroid])

        #tp4 = time.time()
        #print('time4', tp4 - tp3)
        rel_image_path = str(Path(image_path).relative_to(self.image_folder))
        assert len(coords) == len(patches)

        #if self.return_mask:
        return rel_image_path, np.array(coords), np.array(patches)

def object_list(batch):
    return [b for b in zip(*batch)]
    #pickle.dump(batch, open('test.pkl', 'wb'))
    #sys.exit()
    ##res = []
    #path = []
    #coord = []
    #patch = []
    #for p, c, img in batch:
    #    path.append(p)
    #    coord.append(c)
    #    patch.append(img)

    #    res.append([path, coords, patches])


    #print(len(batch))
    #print(batch[0])
    #res = [b for b in zip(*batch)]
    #for path, coords, patches in res:
        #print(path, coords.shape, patches.shape)
    #import sys; sys.exit()
    #return path, coord, patch

class Config:

    def __init__(self):
        self.image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC/'
        self.label_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_su/proto/mask/CRC'
        self.image_patch_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_lmdb/crc/res50_1792_avg_knn/proto/image_patches'
        self.feature_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_lmdb/crc/res50_1792_avg_knn/proto/feature'
        self.patch_size = 64

        self.patch_crop_dataset = ImageNumpyCrop(self.image_path, self.label_path, self.patch_size)

        #mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr
        #std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        #self.valid_transforms = transforms.Compose([
        #    transforms.ToPILImage(),
        #    transforms.Resize((patch_size, patch_size)),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean, std),
        #])
        #self.dataset = ImageNumpy(self.image_path, self.label_path, self.image_size, self.return_mask, transform=valid_transforms)
        res50_path = '/home/baiyu/HGIN/checkpoint/191-best.pth'
        self.extract_func = ExtractorResNet50(res50_path)

class LMDBFeatureWriter:
    def __init__(self, path, dataloader, extractor):
        map_size = 10 << 40
        os.makedirs(path, exist_ok=True)
        self.env = lmdb.open(path, map_size=map_size)
        self.dataloader = dataloader
        self.extractor = extractor

    def write(self):
        #def combined(path, coord):
        #    coord = [str(int(c)) for c in coord]
        #    path + '_nuclei_coord_row_col_' + '_'.join(coord)
        #    return path

        with self.env.begin(write=True) as txn:
            #import time
            #start = time.time()
            #count = 0
            #for rel_pathes, coords, patches in dataloader:
            for image_name, image in dataloader:
                #for path, coord, patch in zip(rel_pathes, coords, patches):
                #    count += 1
                #    for c, p in zip(coord, patch):
                #        c, p
                feature = self.extractor(image)
                print(len(image_name), feature.shape)
                for name, feat in zip(image_name, feature.cpu().numpy()):
                    txn.put(image_name, feature)
                #print(count / (time.time() - start))

class LMDBFeatureReader:
    def __init__(self, path, transform):
        self.env = lmdb.open(path, map_size=1099511627776, readonly=True, lock=False)
        #with self.env.begin(write=False) as txn:
        #    self.image_names= [key.decode() for key in txn.cursor().iternext(keys=True, values=False)]

        #mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr
        #std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        #self.transforms = transforms.Compose([
        #    transforms.ToPILImage(),
        #    #transforms.Resize((self.patch_size, self.patch_size)),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean, std),
        #])

        cache_file = '_cache_' + ''.join(c for c in path if c in string.ascii_letters)
        cache_path = os.path.join(path, cache_file)
        if os.path.isfile(cache_path):
            self.image_names = pickle.load(open(cache_path, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.image_names = [key.decode() for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.image_names, open(cache_path, "wb"))

        self.transform = transform
        self.clusters, self.clusters_names = self.cluster_image_names()

    def split_name(self, image_name):
        rel_path, coord = image_name.split('_nuclei_coord_row_col_')
        coord = [int(c) for c in coord.split('_')]
        return rel_path, coord

    def cluster_image_names(self):
        clusters = {}
        clusters_names = []
        for image_name in self.image_names:
            rel_path, coord = image_name.split('_nuclei_coord_row_col_')

            if rel_path not in clusters:
                clusters[rel_path] = []
                clusters_names.append(rel_path)

            clusters.append(image_name)

        return clusters, clusters_names

    def to_data(self, rel_path, coord, feat):
        #feat = data['feat']
        #coord = data['coord']
        #rel_path, coord = self.split_name(image_name)
        feat = np.concatenate((feat, coord), axis=-1)
        coord = torch.from_numpy(coord).to(torch.float)
        feat = torch.from_numpy(feat).to(torch.float)

        if '1_normal' in rel_path:
            label = 0
        elif '2_low_grade' in rel_path:
            label = 1
        else:
            label = 2

        y = torch.tensor([label], dtype=torch.long)
        data = Data(x=feat, pos=coord, y=y)

        return data

    def __len__(self):
        return self.image_names

    def __getitem__(self, idx):
        #def split_name(image_name):
        #    rel_path, coord = image_name.split('_nuclei_coord_row_col_')
        #    coord = [int(c) for c in coord.split('_')]
        #    return rel_path, coord

        #image_name = self.image_names[idx]
        cluster_name = self.clusters_names[idx]
        clusters = self.clusters[cluster_name]
        feats = []
        coords = []
        with self.env.begin(write=False) as txn:
            for cluster in clusters:
                feat = txn.get(cluster)
                feat = pickle.loads(data)

                _, coord = split_name(cluster)

                feats.append(feat)
                coords.append(coord)

        data = self.to_data(cluster_name, coord, feat)
        data = self.transform(data)
        data = self.knn(data)

        #rel_path, coord = split_name(image_name.decode())

        return cluster_name, data

    def knn(self, data):
        edge_index = radius_graph(data.pos, 100, None, True, 8)
        data.edge_index = edge_index

        return data

class LMDBPatchWriter:
    def __init__(self, path, dataloader):
        map_size = 10 << 40
        os.makedirs(path, exist_ok=True)
        self.env = lmdb.open(path, map_size=map_size)
        self.dataloader = dataloader

    def write(self):
        def combined(path, coord):
            coord = [str(int(c)) for c in coord]
            path = path + '_nuclei_coord_row_col_' + '_'.join(coord)
            return path

        with self.env.begin(write=True) as txn:
            import time
            start = time.time()
            count = 0
            print('writing image patches into LMDB file....')
            #print(dataloader)
            for rel_pathes, coords, patches in self.dataloader:
                for path, coord, patch in zip(rel_pathes, coords, patches):
                    count += 1
                    #print(path, coord.shape, patch.shape)
                    for c, p in zip(coord, patch):
                        #print(c.shape, p.shape)
                        patch_name = combined(path, c)
                        #print(patch_name)
                        p = pickle.dumps(p)
                        txn.put(patch_name.encode(), p)
                print('{}/{}'.format(count, len(self.dataloader) * self.dataloader.batch_size), 'avg:', count / (time.time() - start))


            if count == 100:
                sys.exit()

class LMDBPatchReader:
    def __init__(self, path):
        self.env = lmdb.open(path, map_size=1099511627776, readonly=True, lock=False)
        #with self.env.begin(write=False) as txn:
        #    self.image_names= [key.decode() for key in txn.cursor().iternext(keys=True, values=False)]

        mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr
        std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize((self.patch_size, self.patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        cache_file = '_cache_' + ''.join(c for c in path if c in string.ascii_letters)
        cache_path = os.path.join(path, cache_file)
        if os.path.isfile(cache_path):
            self.image_names = pickle.load(open(cache_path, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.image_names = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.image_names, open(cache_path, "wb"))

    def __len__(self):
        return self.image_names

    def __getitem__(self, idx):
        #def split_name(image_name):
        #    rel_path, coord = image_name.split('_nuclei_coord_row_col_')
        #    coord = [int(c) for c in coord.split('_')]
        #    return rel_path, coord

        image_name = self.image_names[idx]
        with self.env.begin(write=False) as txn:
            data = txn.get(image_name)
            data = pickle.loads(data)

        #rel_path, coord = split_name(image_name.decode())

        return image_name, data


# load network
#####################################################

def network(network_name, num_classes, pretrained):
    if network_name == 'resnet34':
        from model.resnet import resnet34
        net = resnet34(pretrained=pretrained)
        if num_classes != 1000:
            net.reset_num_classes(num_classes)
    elif network_name == 'resnet50':
        from model.resnet import resnet50
        net = resnet50(pretrained=pretrained)
        if num_classes != 1000:
            net.reset_num_classes(num_classes)
    else:
        raise ValueError('network names not suppored')
    return net

class ExtractorResNet50:
    def __init__(self, path):
        self.net = network('resnet50',  4, False)
        print('loading weight file {}...'.format(path))
        self.net.load_state_dict(torch.load(path))
        print('Done.')
        self.net = self.net.cuda()

    def __call__(self, images):
        with torch.no_grad():
            node_features = self.net(images.cuda())
        return node_features

config = Config()
#######################################################

# extract node featrues
######################################################
#dataset = LMDBPatchReader(config.image_patch_path)
#dataloader = DataLoader(dataset, num_workers=4, batch_size=3000, shuffle=False)
#for rel_path, coord, image in dataloader:
#    features





######################################################
#path = 'sldkfk/sdfksdf.itff'
#coord = [33.010, 432.34]
#def combined(path, coord):
#            coord = [str(int(c)) for c in coord]
#            path = path + '_nuclei_coord_row_col_' + '_'.join(coord)
#            print(path)
#            return path
#
#c = combined(path, coord)
#print(c)
#print(c.split('_nuclei_coord_row_col_')[1].split('_'))
#import sys; sys.exit()

#dataset = ImageNumpyCrop(config.image_path, config.label_path, config.patch_size)

#import random
#rel_image_path, coords, patches, bboxes, image = random.choice(dataset)
#for c in coords:
#    image = cv2.circle(image, tuple(c), 3, (0, 200, 0), cv2.FILLED, 6)
#
#for b in bboxes:
#    image = cv2.rectangle(image, tuple(b[:2]), tuple(b[2:]), (255, 0, 0), thickness=6)
#
#
#image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
#cv2.imwrite('/home/baiyu/HGIN/heihei_del11.png', image)
#
#
#
#
#import sys; sys.exit()

# store image patches to LMDB file
############################################
dataset = config.patch_crop_dataset

dataloader = DataLoader(dataset, num_workers=4, batch_size=16, collate_fn=object_list, shuffle=False)
writer = LMDBPatchWriter(config.image_patch_path, dataloader)
print('writing image patches to {}'.format(config.image_patch_path))
writer.write()
#############################################

# evaluting using resnet50
dataset = LMDBPatchReader(config.image_patch_path)
dataloader = DataLoader(dataset, num_workers=4, batch_size=32, shuffle=False)
writer = LMDBFeatureWriter(config.feature_path, dataloader, config.extract_func)
writer.write()

#################################################

# reading features and constructing graph



import sys; sys.exit()
import time

start = time.time()
count = 0
count1 = 0
#for i in dataloader:
#    print(type(i), len(i))
#    import sys;sys.exit()

for rel_image_path, coords, patches in dataloader:
    #count += len(patches)
#for i in dataloader:
    print(coords[0].shape, patches[0].shape)
    print(len(rel_image_path), rel_image_path[0])
    print(len(coords))
    print(len(patches), patches[0][0].shape)
        #print(len(p))
    for p in patches:
        count += len(p)

    print(count / (time.time() - start))
    count1 += len(rel_image_path)
    print()
    print(count1 / (time.time() - start))
