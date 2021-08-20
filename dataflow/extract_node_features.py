import json
import os
from pathlib import Path
from functools import partial
import sys
import random
import pickle
import glob
import string
import time
import csv
#sys.path.append('../')
sys.path.append(os.getcwd())

import lmdb
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import adaptive_avg_pool3d
from torchvision import transforms
#from torch_geometric.data import Data
#from torch_geometric.nn import radius_graph
#from torch_cluster import grid_cluster
#from torch_scatter import scatter
#from torch_geometric.nn.pool.consecutive import consecutive_cluster
#from torch_geometric.nn.pool.pool import pool_pos


import cv2
import numpy as np
from skimage.measure import regionprops


from stich import JsonFolder, JsonDataset, ImageFolder, ImageDataset




class Mask:
    def __init__(self, path):
        self.path = path

    def __init__(self):
        NotImplementedError

    def load_data(self, data_path):
        """return a list of bboxes[min_y, min_x, max_y, max_x]"""
        #min_y = bbox[0]
        #min_x = bbox[1]
        #max_y = bbox[2]
        #max_x = bbox[3]
        NotImplementedError

    def __len__(self):
        NotImplementedError

    def __getitem__(self, image_name):
        image_name = os.path.basename(image_name)
        image_name = image_name.split('.')[0]
        return self.load_data(image_name)

## image --> bbox, coords
class MaskJson(Mask):
    def __init__(self, path):
        self.path = path

#valid_transforms = transforms.Compose([
#        transforms.ToPILImage(),
#        transforms.Resize((64, 64)),
#        transforms.ToTensor(),
#        transforms.Normalize(mean, std),
#    ])

class Patches:
    def __init__(self, image, coords, bboxes, patch_size):
        """
            bboxes :bbox [min_y, min_x, max_y, max_x]
            coords: (row, col)
        """

        #print(len(coords), len(bboxes), coords[10])
        #bboxes, coords = self.filter(bboxes, coords)
        #print(len(coords), len(bboxes), 444444444, coords[10])
        self.image = image
        self.bboxes = bboxes
        self.patch_size = patch_size
        self.coords = coords
        #mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr
        #std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        mean = [0.73646324, 0.56556627, 0.70180897] # Expanuke bgr
        #self.std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        std = [0.18869222, 0.21968669, 0.17277594] # Expanuke bgr


        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.RandomChoice([
            #    transforms.RandomResizedCrop(patch_size, scale=(0.9, 1.1)),
            #    transforms.Resize((patch_size, patch_size))
            #]),
            transforms.RandomApply(torch.nn.ModuleList([
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
            ]), p=1),
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    #def filter(self, bboxes, coords):
    #    res1 = []
    #    res2 = []
    #    for bbox, coord in zip(bboxes, coords):
    #        #if bbox[0] < 0:
    #        #    print(bbox)
    #        #    continue
    #        #if bbox[1] < 0:
    #        #    print(bbox)
    #        #    continue
    #        #if bbox[2] < 0:
    #        #    print(bbox)
    #        #    continue
    #        #if bbox[3] < 0:
    #        #    print(bbox)
    #        #    continue

    #        if bbox[2] - bbox[0] <= 0:
    #            print(bbox, 111)
    #            continue
    #        if bbox[3] - bbox[1] <= 0:
    #            print(bbox, 222)
    #            continue

    #        bbox = [max(b, 0) for b in bbox]
    #        res1.append(bbox)
    #        res2.append(coord)

    #    return res1, res2

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, idx):
        bbox = self.bboxes[idx]
        #print(bbox, idx)
        bbox = self.pad_patch(*bbox, self.patch_size)

        #start_point = (bbox[1], bbox[0])
        #end_point = (bbox[3], bbox[2])
        #self.image = cv2.rectangle(self.image, start_point, end_point, (255, 0, 0), 3)
        #self.image = cv2.circle(self.image, self.coords[idx][::-1], 3, (255, 0, 0), 3)


        patch = self.image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
        #print(patch.shape, idx)
        #try:
        patch = self.transforms(patch)
        #except:
        #    print('wrong image:', patch.shape, bbox)
        #    cv2.imwrite('ffff.jpg', patch)
        #    cv2.imwrite('xxxx.jpg', self.image)
        #    sys.exit()
        return patch, np.array(self.coords[idx])

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

#### generating features
class ExtractorResNet50:
    def __init__(self, path):
        self.net = network('resnet50',  5, False)
        print('loading weight file {}...'.format(path))
        self.net.load_state_dict(torch.load(path))
        print('Done.')
        self.net = self.net.cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=[0, 1, 2])

    def __call__(self, images):
        with torch.no_grad():
            output = self.net(images.cuda())
            output = output.unsqueeze(0)
            output = adaptive_avg_pool3d(output, (512, 1, 1))
            output = output.squeeze()
        return output.cpu().numpy()

#def extractor_resnet(images, mask=None):
    #net = network('resnet50',  4, False)
    #net = net.load_state_dict(torch.load(res50_path))
    #node_features = net(images)
    #return node_features

#class ExtractNodeFeat:
#    def __init__(self, patch_size, extractor):
#        self.dataset = []
#        self.extractor = extractor
#        self.length = []
#        self.patch_size = patch_size
#        self.path = []
#        self.mask = []
#
#    def update(self, image, path, bboxes, coords, mask):
#        assert len(bboxes) == len(coords)
#        dataset = Patches(image, coords, bboxes, self.patch_size)
#        self.dataset.append(dataset)
#        self.length.append(len(bboxes))
#        self.path.append(path)
#        self.mask.append(mask)
#
#    def run(self, batch_size):
#        dataset = torch.utils.data.ConcatDataset(self.dataset)
#        data_loader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=False, collate_fn=object_list)
#
#        node_features = []
#        node_coords = []
#        #with torch.no_grad():
#        with torch.no_grad():
#            for images, coords in data_loader:
#                images = torch.stack(images)
#                print(images.shape)
#                output = self.extractor(images)
#                output = output.unsqueeze(0)
#                output = adaptive_avg_pool3d(output, (16, 1, 1))
#                output = output.squeeze()
#                node_features.append(output.cpu().numpy())
#                node_coords.extend(coords)
#
#        if node_features:
#            node_features = np.vstack(node_features)
#        else:
#            node_features = np.array(node_features)
#
#        if node_coords:
#            node_coords = np.vstack(node_coords)
#        else:
#            node_coords = np.array(node_coords)
#
#        assert len(node_coords) == len(node_features)
#
#        feats = []
#        coords = []
#        prev = 0
#        for l in self.length:
#            feats[prev:prev + l]
#            coords[prev:prev + l]
#            prev += l
#
#        assert prev == sum(self.length)
#
#        self.dataset = []
#        self.length = []
#        path = self.path
#        self.path = []
#        mask = self.mask
#        self.mask = []
#
#        assert len(feats) == len(coords) == len(path) == len(mask)
#
#        return feats, coords, path, mask




#def extract_node_features_resnet(image, bboxes, coords, patch_size, batch_size, extractor):
#    import time
#    start = time.time()
#    dataset = Patches(image, coords, bboxes, patch_size)
#    data_loader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=False)
#
#    node_features = []
#    node_coords = []
#    #with torch.no_grad():
#    with torch.no_grad():
#        for images, coords in data_loader:
#        #coords = []
#        #for image, coord in dataset:
#            #node_features = net(images)
#            output = extractor(images)
#            output = output.unsqueeze(0)
#            output = adaptive_avg_pool3d(output, (16, 1, 1))
#            output = output.squeeze()
#
#            node_features.append(output.cpu().numpy())
#
#            node_coords.append(coords.numpy())
#
#
#
#        if node_features:
#            node_features = np.vstack(node_features)
#        else:
#            node_features = np.array(node_features)
#
#        if node_coords:
#            node_coords = np.vstack(node_coords)
#        else:
#            node_coords = np.array(node_coords)
#
#        print(len(node_coords))
#        assert len(node_coords) == len(node_features)
#
#    finish = time.time()
#    print(finish - start)
#    return node_features, node_coords


#def extract_node_features_resnet(image, bboxes, coords, patch_size, batch_size, extractor):
#    import time
#    start = time.time()
#    dataset = Patches(image, coords, bboxes, patch_size)
#    data_loader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=False)
#
#    node_features = []
#    node_coords = []
#    #with torch.no_grad():
#    with torch.no_grad():
#        #for images, coords in data_loader:
#        images = []
#        #coords = []
#        count = 0
#        for image, coord in dataset:
#            images.append(image)
#            node_coords.append(coord)
#
#
#            #node_features = net(images)
#            count += 1
#            if count == batch_size:
#                images = torch.stack(images)
#                output = extractor(images)
#                output = output.unsqueeze(0)
#                output = adaptive_avg_pool3d(output, (16, 1, 1))
#                output = output.squeeze()
#
#                node_features.append(output.cpu().numpy())
#
#                #node_coords.append(coords.numpy())
#                images = []
#                count = 0
#
#
#        if images:
#                #images
#                images = torch.stack(images)
#                output = extractor(images)
#                output = output.unsqueeze(0)
#                output = adaptive_avg_pool3d(output, (16, 1, 1))
#                output = output.squeeze()
#
#                node_features.append(output.cpu().numpy())
#
#                #node_coords.append(coords.numpy())
#
#        if node_features:
#            node_features = np.vstack(node_features)
#        else:
#            node_features = np.array(node_features)
#
#        if node_coords:
#            node_coords = np.vstack(node_coords)
#        else:
#            node_coords = np.array(node_coords)
#
#        assert len(node_coords) == len(node_features)
#
#    finish = time.time()
#    print(finish - start)
#    return node_features, node_coords


#def extract_node_features_hand_crafted(image, bboxes, patch_size, batch_size, extractor):
#    pass
#
#class ImageNumpy:
#    def __init__(self, image_folder, label_folder, image_size, return_mask=False):
#        self.image_folder = image_folder
#        self.label_folder = label_folder
#        self.image_names = []
#        self.return_mask = return_mask
#        search_path = os.path.join(image_folder, '**', '*.png')
#        for image_name in glob.iglob(search_path, recursive=True):
#            if 'mask' in image_name:
#                continue
#            self.image_names.append(image_name)
#
#    def __len__(self):
#        return len(self.image_names)
#
#    def image2mask_fp(self, image_path):
#        rel_image_path = Path(image_path).relative_to(self.image_folder)
#        mask_path = os.path.join(self.label_folder, str(rel_image_path).replace('.png', '.npy'))
#        return mask_path
#
#    def __getitem__(self, idx):
#        image_path = self.image_names[idx]
#        image = cv2.imread(image_path)
#        image = cv2.resize(image, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
#
#        bboxes = []
#        coords = []
#
#        mask_path = self.image2mask_fp(image_path)
#        mask = np.load(mask_path)
#        props = regionprops(mask)
#        for prop in props:
#            bboxes.append([int(c) for c in prop.bbox])
#            coords.append([int(c) for c in prop.centroid])
#
#        rel_image_path = str(Path(image_path).relative_to(self.image_folder))
#        if self.return_mask:
#            return image, rel_image_path, bboxes, coords, mask
#        else:
#            return image, rel_image_path, bboxes, coords, None

#class ImageJson:
#    def __init__(self, image_folder, label_folder, image_size, return_mask=False):
#        self.image_folder = image_folder
#        self.image_size = image_size
#        self.label_folder = label_folder
#        json_lists = JsonFolder(label_folder)
#        self.json_dataset = JsonDataset(json_lists, image_size)
#        image_list = ImageFolder(image_folder)
#        self.image_dataset = ImageDataset(image_list, image_size)
#        self.image_lists = self.image_dataset.file_path_lists
#        self.return_mask = return_mask
#        #self.json_lists = self.json_dataset.file_path_lists
#        #self.transform = transform
#
#    def image_prefixes(self):
#        return self.image_dataset.file_prefixes
#
#    def image_lists_by_prefix(self, prefix):
#        return self.image_dataset.file_grids[prefix]
#
#    def get_res_by_image_path(self, image_path):
#        image_path = Path(image_path)
#        rel_image_path = str(image_path.relative_to(self.image_folder))
#        image = self.image_dataset.get_file_by_path(str(image_path))
#        image = cv2.resize(image, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
#        json_path = self.image2json_fp(image_path)
#        labels = self.json_dataset.get_file_by_path(json_path)
#        #print(type(labels))
#        bboxes, coords, mask = self.format_labels(labels)
#        return rel_image_path, image, bboxes, coords, mask
#
#
#    def __getitem__(self, idx):
#        image_path = Path(self.image_lists[idx])
#        rel_image_path = str(image_path.relative_to(self.image_folder))
#        image = self.image_dataset.get_file_by_path(str(image_path))
#        image = cv2.resize(image, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
#        json_path = self.image2json_fp(image_path)
#        labels = self.json_dataset.get_file_by_path(json_path)
#        #print(type(labels))
#        bboxes, coords, mask = self.format_labels(labels)
#        return image, rel_image_path, bboxes, coords, mask
#
#    def __len__(self):
#        assert len(self.image_dataset) == len(self.json_dataset)
#        return len(self.image_dataset)
#
#    def image2json_fp(self, image_path):
#        base_name = os.path.basename(image_path)
#        json_file = base_name.split('.')[0] + '.json'
#        return os.path.join(self.label_folder, json_file)
#
#    def format_labels(self, labels):
#        coords = []
#        bboxes = []
#        if self.return_mask:
#            mask = np.zeros((self.image_size, self.image_size), dtype='uint8')
#
#        #print(labels.keys())
#        for node in labels:
#            cen = node['centroid']
#            cen = [int(c) for c in cen]
#            cen = cen[::-1]
#            coords.append(cen)
#            #image = cv2.circle(image, tuple(cen), 3, (0, 200, 0), cv2.FILLED, 1)
#            # node['bbox'] : [[rmin, cmin], [rmax, cmax]]
#            bbox = node['bbox']
#            # bbox : [min_y, min_x, max_y, max_x]
#            #bbox = [b // 2 for b in sum(bbox, [])]
#            bbox = [b for b in sum(bbox, [])]
#            bboxes.append(bbox)
#
#            #contour = node['contour']
#            #node['contour'] = [[c1 // 2, c2 // 2] for [c1, c2] in contour]
#
#            if self.return_mask:
#                #cv2.imwrite('heihei_del1.png', image)
#                cv2.drawContours(mask, [np.array(node['contour'])], -1, 255, -1)
#                #cv2.imwrite('heihei_del.png', image)
#                #cv2.imwrite()
#
#        if self.return_mask:
#            return bboxes, coords, mask
#        else:
#            return bboxes, coords, None

#class LMDBWriter:
#    def __init__(self, save_path):
#        self.datasets = {}
#        self.save_path = save_path
#
#    def __call__(self, key, value, printable=False):
#        """key: relative path to image"""
#        assert type(key) == str
#        assert type(value) == bytes
#
#        #image_name = os.path.basename(key)
#        dir_name = os.path.dirname(key)
#
#        if dir_name not in self.datasets:
#            map_size = 10 << 40
#            lmdb_path = os.path.join(self.save_path, dir_name)
#            os.makedirs(lmdb_path, exist_ok=True)
#            env = lmdb.open(lmdb_path, map_size=map_size)
#            self.datasets[dir_name] = env
#
#        env = self.datasets[dir_name]
#        with env.begin(write=True) as txn:
#            if printable:
#                print('writing files to {}...'.format(
#                    os.path.join(self.save_path, key)))
#            txn.put(key.encode(), value)

#class LMDBReader:
#    def __init__(self, path, transform):
#        self.env = lmdb.open(path, map_size=1099511627776, readonly=True, lock=False)
#        #with self.env.begin(write=False) as txn:
#        #    self.image_names= [key.decode() for key in txn.cursor().iternext(keys=True, values=False)]
#
#        cache_file = '_cache_' + ''.join(c for c in path if c in string.ascii_letters)
#        cache_path = os.path.join(path, cache_file)
#        if os.path.isfile(cache_path):
#            self.image_names = pickle.load(open(cache_path, "rb"))
#        else:
#            with self.env.begin(write=False) as txn:
#                self.image_names = [key for key in txn.cursor().iternext(keys=True, values=False)]
#            pickle.dump(self.image_names, open(cache_path, "wb"))
#
#        self.transform = transform
#
#    @classmethod
#    def init_dataset(cls, save_path, transform):
#        datasets = []
#        for lmdb_fp in glob.iglob(os.path.join(save_path, '**', 'data.mdb'), recursive=True):
#            #print(lmdb_fp)
#            datasets.append(cls(os.path.dirname(lmdb_fp), transform=transform))
#
#        return torch.utils.data.ConcatDataset(datasets)
#
#    def __len__(self):
#        return len(self.image_names)
#
#    def knn(self, data):
#        edge_index = radius_graph(data.pos, 100, None, True, 8)
#        data.edge_index = edge_index
#
#        return data
#
#    def to_data(self, image_name, data):
#        feat = data['feat']
#        coord = data['coord']
#        feat = np.concatenate((feat, coord), axis=-1)
#        coord = torch.from_numpy(coord).to(torch.float)
#        feat = torch.from_numpy(feat).to(torch.float)
#
#        if '1_normal' in image_name:
#            label = 0
#        elif '2_low_grade' in image_name:
#            label = 1
#        else:
#            label = 2
#
#        y = torch.tensor([label], dtype=torch.long)
#        data = Data(x=feat, pos=coord, y=y)
#
#        return data
#    #def __getitem__(self, idx):
#    #    NotImplementedError
#    def __getitem__(self, idx):
#
#        image_name = self.image_names[idx]
#        with self.env.begin(write=False) as txn:
#            data = txn.get(image_name)
#            data = pickle.loads(data)
#
#        data = self.to_data(image_name.decode(), data)
#        if self.transform:
#            data = self.transform(data)
#
#        data = self.knn(data)
#
#        return image_name.decode(), data


#def lmdb_concatenate(save_path, transform=None):
#    datasets = []
#    for lmdb_fp in glob.iglob(os.path.join(save_path, '**', 'data.mdb'), recursive=True):
#        #print(lmdb_fp)
#        datasets.append(LMDBReader(os.path.dirname(lmdb_fp), transform=transform))
#
#    return torch.utils.data.ConcatDataset(datasets)


#def _read_one_raw_graph( raw_file_path):
#    # import pdb;pdb.set_trace()
#    nodes_features = np.load( raw_file_path + '.npy')
#    coordinates = np.load(raw_file_path.replace('feature', 'coordinate') + '.npy')
#    nodes_features = np.concatenate((nodes_features, coordinates), axis= -1)
#    coordinates = torch.from_numpy(coordinates).to(torch.float)
#    nodes_features = torch.from_numpy(nodes_features ).to(torch.flo, uat)
#    if '1_normal' in raw_file_path:
#        label = 0
#    elif '2_low_grade' in raw_file_path:
#        label = 1
#    else:
#        label = 2
#    y = torch.tensor([label], dtype=torch.long)
#    data = Data(x = nodes_features, pos = coordinates, y = y)
#    return data

#def _avg_pool_x(cluster, x, size=None):
#    return scatter(x, cluster, dim=0, dim_size=size, reduce='mean')
#
#def avg_pooling(data):
#    cluster = grid_cluster(data.pos, torch.Tensor([64, 64]))
#    cluster, perm = consecutive_cluster(cluster)
#    x = None if data.x is None else _avg_pool_x(cluster, data.x)
#    pos = None if data.pos is None else pool_pos(cluster, data.pos)
#    data.x = x
#    data.pos = pos
#
#    return data
#
#def _add_pool_x(cluster, x, size=None):
#    return scatter(x, cluster, dim=0, dim_size=size, reduce='add')
#
#def add_pooling(data):
#    cluster = grid_cluster(data.pos, torch.Tensor([64, 64]))
#    cluster, perm = consecutive_cluster(cluster)
#    x = None if data.x is None else _add_pool_x(cluster, data.x)
#    pos = None if data.pos is None else pool_pos(cluster, data.pos)
#    data.x = x
#    data.pos = pos
#
#    return data

#def object_list(batch):
#    #pathes = []
#    #data = []
#    #for p, d in batch:
#    #    pathes.append(p)
#    #    data.append(d)
#
#    #print(len(batch))
#    #res = [b for b in batch]
#    #print(len(res))
#    #return pathes, data
#    return [b for b in zip(*batch)]

#def object_list(batch):

#def gen_training_data(image_folder, label_folder, extract_func, feature_type, image_size, save_path):
#def gen_training_data(conf):
#    """type: hand-crafted, resnet50"""
#
#    dataset = conf.dataset
#    #if feature_type == 'hand':
#    #    extractor = extract_node_features_hand_crafted
#    #    dataset = None
#    #elif feature_type == 'res50':
#    #    extractor = extract_node_features_resnet
#    #    dataset = ImageJson(image_folder, label_folder, image_size)
#
#    #####processed data
#
#    #image, rel_image_path, bboxes, coords, mask = random.choice(dataset)
#    data_loader = DataLoader(dataset, num_workers=2, batch_size=16, shuffle=False, collate_fn=object_list)
#    print('extracting node features......')
#    count = 0
#    for image, rel_image_path, bboxes, coords, mask in data_loader:
#
#        for img, p, box, coord, m in zip(image, rel_image_path, bboxes, coords, mask):
#
#        #    count += 1
#        #    if count % 10 != 0:
#        #        print(conf.node_feature_coords)
#        #        conf.node_feature_coords.update(img, p, box, coord, m)
#        #        continue
#
#
#        #print(bboxes)
#    #for coord in coords:
#    #        image = cv2.circle(image, tuple(coord[::-1]), 3, (0, 200, 0), cv2.FILLED, 3)
#
#    #image = cv2.resize(image, (0, 0), fx=0.125, fy=0.125)
#    #cv2.imwrite('/home/baiyu/HGIN/heihei_del11.png', image)
#    #sys.exit()
#    #for bbox in bboxes:
#    #    image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (255, 0, 0), thickness=1)
#
#    ##image = cv2.drawContours(image, [np.array(node['contour']) // 2 ], -1, (0, 0, 0), -1)
#
#    #image[mask != 0] = 0
#    #cv2.imwrite('heihei_del11.png', image)
#    #sys.exit()
#    ############ saving node_features and node_coords
#            node_features, node_coords = conf.node_feature_coords(img, box, coord)
#            #node_features, node_coords, path, mask = conf.node_feature_coords.run(3000)
#
#            #for node_feat, node_coord, p, m in zip(node_features, node_coords, path, mask):
#
#            res = {
#                'feat': node_features,
#                'coord' : node_coords,
#                #'mask' : mask
#                'mask' : m
#            }
#
#            val = pickle.dumps(res)
#            count += 1
#            print('[{}/{}]'.format(count, len(dataset)))
#            conf.writer(str(p), val, conf.print)
#
#    ######################## read features
#    print('generating cell graph....')
#    dataset = conf.reader(conf.save_path)
#    save_path = conf.training_data_path
#    #from torch_geometric.data import DataLoader
#
#    data_loader = DataLoader(dataset, num_workers=4, batch_size=128, shuffle=False, collate_fn=object_list)
#
#    for epoch in range(conf.epoches):
#        for path, data in data_loader:
#
#            for p, d in zip(path, data):
#                os.makedirs(os.path.join(save_path, str(epoch)), exist_ok=True)
#                fp = os.path.join(save_path, str(epoch), p.split('.')[0] + '.pt')
#                print(fp)
                #torch.save(d, fp)



    #for image_name, val in dataset:
        #print(image_name)
        #image_name.
        #for k, v in val.items():
            #if k == 'mask':
                #continue
            #print(k, v.shape)


    #sys.exit()


    #for coord in node_coords:
    #    image = cv2.circle(image, tuple(coord[::-1]), 3, (0, 200, 0), cv2.FILLED, 1)

    #cv2.imwrite('heihei_del11.png', image)

#class Res50BaseConfig:
#    def __init__(self):
#        self.epoches = 1
#        self.image_size = 224
#        self.print = True
#
#        self.image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC/'
#        self.label_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_su/proto/mask/CRC'
#        self.save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data/crc/res50_1792_avg_knn/proto/'
#        self.pool = 'avg'
#        transform = avg_pooling
#        patch_size = 64
#        #self.pool = 'add':
#        #transform = add_pooling
#
#        self.method = 'knn'
#        self.seg = 'cia'
#        folder_name = 'fix_{}_{}_{}'.format(self.pool, self.seg, self.method)
#        self.training_data_path = os.path.join('/home/baiyu/training_data/CRC', folder_name)
#        self.return_mask = True
#        #res50_path = '/home/baiyu/HGIN/checkpoint/191-best.pth'
#        #res50_path = 'checkpoint/resnet50/Tuesday_01_June_2021_19h_43m_05s/191-best.pth' # colon
#        res50_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ResNet50/97-best.pth'
#
#        #mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr
#        #std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
#        #self.valid_transforms = transforms.Compose([
#        #    transforms.ToPILImage(),
#        #    transforms.Resize((patch_size, patch_size)),
#        #    transforms.ToTensor(),
#        #    transforms.Normalize(mean, std),
#        #])
#        #self.dataset = ImageNumpy(self.image_path, self.label_path, self.image_size, self.return_mask, transform=valid_transforms)
#        self.extract_func = ExtractorResNet50(res50_path)
#        #self.writer = LMDBWriter(self.save_path)
#        #self.reader = partial(LMDBReader.init_dataset, transform=transform)
#        #self.node_feature_coords = ExtractNodeFeat(patch_size=patch_size,
#        #                                                    extractor=self.extract_func)
#        #extract_node_feat
#        self.node_feature_coords = partial(extract_node_features_resnet,
#                                        batch_size=3000,
#                                        patch_size=patch_size,
#                                        extractor=self.extract_func,
#                                        )

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


#class Res50JsonMaskConfig(Res50BaseConfig):
#    def __init__(self):
#        super().__init__()
#        self.image_path = '/data/by/tmp/HGIN/test_can_be_del3'
#        self.label_path = '/data/by/tmp/HGIN/test_can_be_del2/EXtended_CRC_Mask'
#        #self.label_path = '/home/baiyu/EXtended_CRC_Mask'
#        self.save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_lmdb/extended_crc/feat/'
#
#        self.pool = 'avg'
#        transform = avg_pooling
#        #self.pool = 'add'
#        #transform = add_pooling
#
#        self.seg = 'hover'
#        folder_name = 'fix_{}_{}_{}'.format(self.pool, self.seg, self.method)
#        self.training_data_path = os.path.join('/home/baiyu/training_data/ExtendedCRC', folder_name)
#        self.return_mask = False
#        #self.dataset = ImageJson(self.image_path, self.label_path, self.image_size, self.return_mask)
#        #self.dataset = ImageJson(self.image_path, self.label_path, self.image_size, self.return_mask)
#        #self.reader = lmdb_concatenate(self.save_path, transform=transform)
#        #self.writer = LMDBWriter(self.save_path)
#        #self.reader = partial(LMDBReader.init_dataset, transform=transform)

class FeatLMDBWriter:
    def __init__(self, save_path):
        map_size = 10 << 40
        self.env = lmdb.open(save_path, map_size=map_size)

    def add_pair(self, pairs):
        with self.env.begin(write=True) as txn:
            for pair in pairs:
                name, val = pair
                txn.put(name.encode(), pickle.dumps(val))

    #def __call__(self, key, value, printable=False):
    #    """key: relative path to image"""
    #    assert type(key) == str
    #    assert type(value) == bytes

    #    #image_name = os.path.basename(key)
    #    dir_name = os.path.dirname(key)

    #    if dir_name not in self.datasets:
    #        lmdb_path = os.path.join(self.save_path, dir_name)
    #        os.makedirs(lmdb_path, exist_ok=True)
    #        env = lmdb.open(lmdb_path, map_size=map_size)
    #        self.datasets[dir_name] = env

    #    env = self.datasets[dir_name]
    #    with env.begin(write=True) as txn:
    #        if printable:
    #            print('writing files to {}...'.format(
    #                os.path.join(self.save_path, key)))
    #        txn.put(key.encode(), value)

class BACH:
    def __init__(self, image_folder, json_folder):
        #test_names, test_labels = self.read_csv(test_csv)
        #self.image_names = test_names
        #self.test_labels = test_labels

        self.image2path = {}
        self.json2path = {}
        search_path = os.path.join(image_folder, '**', '*.tif')
        for image_fp in glob.iglob(search_path, recursive=True):
            image_name = os.path.basename(image_fp)
            prefix = image_name.split('.')[0]
            self.image2path[prefix] = image_fp

        search_path = os.path.join(json_folder, '**', '*.json')
        for json_fp in glob.iglob(search_path, recursive=True):
            json_name = os.path.basename(json_fp)
            prefix = json_name.split('.')[0]
            self.json2path[prefix] = json_fp

        self.image_names = list(self.image2path.keys())

    #def read_csv(self, csv_fp):
    #    image_names = []
    #    labels = []
    #    with open(csv_fp) as csvfile:
    #        reader = csv.DictReader(csvfile)
    #        for row in reader:
    #            image_name, label = row['image name'], row['gleason score']
    #            image_names.append(image_name)
    #            #labels.append(label)
    #            cls_id = int(label) - 5
    #            labels.append(cls_id)

    #    return image_names, label

    def __len__(self):
        return len(self.image_names)

    def read_json(self, json_path):
        #res = []
        coords = []
        bboxes = []

        with open(json_path, 'r') as f:
            json_data = json.load(f)
            #print(res.keys())
            #print(res['nuc']['2175'])
            #print(len(res['nuc'].keys()))
            for k, node in json_data['nuc'].items():
                #print(type(v))
                #print(v.keys())
                cen = node['centroid']
                cen = [int(c) for c in cen]
                cen = cen[::-1]
                coords.append(cen)
                #image = cv2.circle(image, tuple(cen), 3, (0, 200, 0), cv2.FILLED, 1)
                # node['bbox'] : [[rmin, cmin], [rmax, cmax]]
                bbox = node['bbox']
                # bbox : [min_y, min_x, max_y, max_x]
                #bbox = [b // 2 for b in sum(bbox, [])]
                bbox = [b for b in sum(bbox, [])]
                bboxes.append(bbox)
                #res.append(v)

        return bboxes, coords

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        #print(image_name)
        prefix = image_name.split('.')[0]
        #print(self.image2path.keys())
        image_fp = self.image2path[prefix]
        image = cv2.imread(image_fp, -1)

        json_fp = self.json2path[prefix]
        #print(image_fp, json_fp)
        bboxes, coords = self.read_json(json_fp)

        return os.path.basename(image_fp), image, bboxes, coords

class Prostate5CropsAug:
    def __init__(self, image_folder, json_folder):
        #test_names, test_labels = self.read_csv(test_csv)
        #self.image_names = test_names
        #self.test_labels = test_labels

        self.image2path = {}
        self.json2path = {}
        search_path = os.path.join(image_folder, '**', '*.jpg')
        for image_fp in glob.iglob(search_path, recursive=True):
            image_name = os.path.basename(image_fp)
            prefix = image_name.split('.')[0]
            self.image2path[prefix] = image_fp

        search_path = os.path.join(json_folder, '**', '*.json')
        for json_fp in glob.iglob(search_path, recursive=True):
            json_name = os.path.basename(json_fp)
            prefix = json_name.split('.')[0]
            self.json2path[prefix] = json_fp

        self.image_names = list(self.image2path.keys())
        #self.image_names = list(self.image2path.keys())

    #def read_csv(self, csv_fp):
    #    image_names = []
    #    labels = []
    #    with open(csv_fp) as csvfile:
    #        reader = csv.DictReader(csvfile)
    #        for row in reader:
    #            image_name, label = row['image name'], row['gleason score']
    #            image_names.append(image_name)
    #            #labels.append(label)
    #            cls_id = int(label)
    #            if cls_id <= 6 :
    #                labels.append(1)
    #            elif cls_id > 6:
    #                labels.append(2)
    #            else:
    #                raise ValueError('wrong label')

    #            #labels.append(cls_id)

    #    return image_names, labels

    def __len__(self):
        return len(self.image_names)

    def read_json(self, json_path):
        #res = []
        coords = []
        bboxes = []

        with open(json_path, 'r') as f:
            json_data = json.load(f)
            #print(res.keys())
            #print(res['nuc']['2175'])
            #print(len(res['nuc'].keys()))
            for k, node in json_data['nuc'].items():
                #print(type(v))
                #print(v.keys())
                cen = node['centroid']
                cen = [int(c) for c in cen]
                cen = cen[::-1]
                coords.append(cen)
                #image = cv2.circle(image, tuple(cen), 3, (0, 200, 0), cv2.FILLED, 1)
                # node['bbox'] : [[rmin, cmin], [rmax, cmax]]
                bbox = node['bbox']
                # bbox : [min_y, min_x, max_y, max_x]
                #bbox = [b // 2 for b in sum(bbox, [])]
                bbox = [b for b in sum(bbox, [])]
                bboxes.append(bbox)
                #res.append(v)

        return bboxes, coords

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        #print(image_name)
        #prefix = image_name.split('.')[0]
        prefix = image_name
        #print(len(self.image2path.keys()))
        image_fp = self.image2path[prefix]
        image = cv2.imread(image_fp, -1)

        json_fp = self.json2path[prefix]
        #print(image_fp, json_fp)
        bboxes, coords = self.read_json(json_fp)

        base_name = os.path.basename(image_fp).replace('_aug_', '_aug_jitter')
        #dir_name = os.path.dirname(image_fp)
        #label = self.mask_labels[idx]
        #base_name = base_name.replace('.', '_grade_{}.'.format(label))

        return  base_name, image, bboxes, coords

class Prostate5Crops:
    def __init__(self, image_folder, json_folder, test_csv):
        #test_names, test_labels = self.read_csv(test_csv)
        #self.image_names = test_names
        #self.test_labels = test_labels

        self.image2path = {}
        self.json2path = {}
        search_path = os.path.join(image_folder, '**', '*.jpg')
        for image_fp in glob.iglob(search_path, recursive=True):
            image_name = os.path.basename(image_fp)
            prefix = image_name.split('.')[0]
            self.image2path[prefix] = image_fp

        search_path = os.path.join(json_folder, '**', '*.json')
        for json_fp in glob.iglob(search_path, recursive=True):
            json_name = os.path.basename(json_fp)
            prefix = json_name.split('.')[0]
            self.json2path[prefix] = json_fp

        self.image_names, self.mask_labels = self.read_csv(test_csv)
        #self.image_names = list(self.image2path.keys())

    def read_csv(self, csv_fp):
        image_names = []
        labels = []
        with open(csv_fp) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image_name, label = row['image name'], row['gleason score']
                image_names.append(image_name)
                #labels.append(label)
                cls_id = int(label)
                if cls_id <= 6 :
                    labels.append(1)
                elif cls_id > 6:
                    labels.append(2)
                else:
                    raise ValueError('wrong label')

                #labels.append(cls_id)

        return image_names, labels

    def __len__(self):
        return len(self.image_names)

    def read_json(self, json_path):
        #res = []
        coords = []
        bboxes = []

        with open(json_path, 'r') as f:
            json_data = json.load(f)
            #print(res.keys())
            #print(res['nuc']['2175'])
            #print(len(res['nuc'].keys()))
            for k, node in json_data['nuc'].items():
                #print(type(v))
                #print(v.keys())
                cen = node['centroid']
                cen = [int(c) for c in cen]
                cen = cen[::-1]
                coords.append(cen)
                #image = cv2.circle(image, tuple(cen), 3, (0, 200, 0), cv2.FILLED, 1)
                # node['bbox'] : [[rmin, cmin], [rmax, cmax]]
                bbox = node['bbox']
                # bbox : [min_y, min_x, max_y, max_x]
                #bbox = [b // 2 for b in sum(bbox, [])]
                bbox = [b for b in sum(bbox, [])]
                bboxes.append(bbox)
                #res.append(v)

        return bboxes, coords

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        #print(image_name)
        prefix = image_name.split('.')[0]
        #print(len(self.image2path.keys()))
        image_fp = self.image2path[prefix]
        image = cv2.imread(image_fp, -1)

        json_fp = self.json2path[prefix]
        #print(image_fp, json_fp)
        bboxes, coords = self.read_json(json_fp)

        base_name = os.path.basename(image_fp)
        #dir_name = os.path.dirname(image_fp)
        label = self.mask_labels[idx]
        base_name = base_name.replace('.', '_grade_{}.'.format(label))

        return  base_name, image, bboxes, coords


class Prosate:
    def __init__(self, image_folder, json_folder):
        self.image2path = {}
        self.json2path = {}
        search_path = os.path.join(image_folder, '**', '*.jpg')
        for image_fp in glob.iglob(search_path, recursive=True):
            image_name = os.path.basename(image_fp)
            self.image2path[image_name] = image_fp

        search_path = os.path.join(json_folder, '**', '*.json')
        for json_fp in glob.iglob(search_path, recursive=True):
            json_name = os.path.basename(json_fp)
            self.json2path[json_name] = json_fp

        self.image_names = list(self.image2path.keys())

    def __len__(self):
        return len(self.image2path.keys())

    def read_json(self, json_path):
        #res = []
        coords = []
        bboxes = []

        with open(json_path, 'r') as f:
            json_data = json.load(f)
            #print(res.keys())
            #print(res['nuc']['2175'])
            #print(len(res['nuc'].keys()))
            for k, node in json_data['nuc'].items():
                #print(type(v))
                #print(v.keys())
                cen = node['centroid']
                cen = [int(c) for c in cen]
                cen = cen[::-1]
                coords.append(cen)
                #image = cv2.circle(image, tuple(cen), 3, (0, 200, 0), cv2.FILLED, 1)
                # node['bbox'] : [[rmin, cmin], [rmax, cmax]]
                bbox = node['bbox']
                # bbox : [min_y, min_x, max_y, max_x]
                #bbox = [b // 2 for b in sum(bbox, [])]
                bbox = [b for b in sum(bbox, [])]
                bboxes.append(bbox)
                #res.append(v)

        return bboxes, coords


    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_fp = self.image2path[image_name]
        image = cv2.imread(image_fp, -1)

        json_fp = self.json2path[image_name.replace('.jpg', '.json')]
        #print(image_fp, json_fp)
        bboxes, coords = self.read_json(json_fp)

        return os.path.basename(image_fp), image, bboxes, coords

def image2pt(rel_path):
    rel_path = rel_path.replace('.jpg', '.pt')
    rel_path = rel_path.replace('.png', '.pt')
    return rel_path

def generate_features(dataloader, num_feats, rel_pathes, data_writer, extractor):
    node_features = []
    node_coords = []
    #wih torch.no_grad():
    print('extracting features.....')
    for idx, (images, coords) in enumerate(data_loader):
        output = extractor(images)
        node_features.append(output)
        node_coords.extend(coords)
        print('iter: [{}] / [{}]'.format(idx, len(data_loader)))
    print('done....')

    node_features = np.vstack(node_features)
    node_coords = np.vstack(node_coords)
    prev = 0
    #print('here...........................................')
    #print(len(rel_pathes), len(num_feats))
    #test_cum = 0
    data_pair = []
    for rel_path, num_feat in zip(rel_pathes, num_feats):
        #print(rel_path, num_feat)

        sub_feat = node_features[prev : prev+num_feat]
        sub_coord = node_coords[prev : prev+num_feat]
        #print(sub_feat.shape, sub_coord.shape)
        val = {
            'feat' : sub_feat,
            'coord' : sub_coord
        }
        #val = pickle.dumps(val)
        #rel_path = rel_path.replace('.jpg', '.pt')
        #rel_path = rel_path.replace('.png', '.pt')
        rel_path = image2pt(rel_path)
        #print(rel_path, sub_feat.shape, sub_coord.shape)
        prev += num_feat
        data_pair.append([rel_path, val])

    print('wrtting to disk.....')
    data_writer.add_pair(data_pair)
    print('done.....')
    #print(count / (time.time() - start))



class TorchWriter:
    def __init__(self, save_path):
        self.save_path = save_path
    def add_pair(self, pairs):
        #import time
        #start = time.time()
        #saves = 0
        #to_tensors = 0
        for idx, pair in enumerate(pairs):
            #load = time.time()
            name, val = pair
            for k, v in val.items():
                val[k] = torch.tensor(v)
            path = os.path.join(self.save_path, name)
            #to_tensor = time.time()
            #print(path)

            print(path, idx)
            torch.save(val, path)
            #import sys; sys.exit()

            #save = time.time()

            #saves += save - to_tensor
            #to_tensors += to_tensor - load
            #loads = save - saves - to_tensors

            #total = save - start
            #print(loads / total, to_tensors / total, saves / total)




if __name__ == '__main__':
    print()

    #save_path = ''
    #dataset = ImageJson('/home/baiyu/Extended_CRC', '/home/baiyu/EXtended_CRC_Mask', 1792)
    #image_path = '/home/baiyu/Extended_CRC'
    #json_path = '/home/baiyu/EXtended_CRC_Mask'
    #config = Res50JsonMaskConfig()
    resnet50_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ResNet50/97-best.pth'
    #image_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images'
    #json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat/'
    #image_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images_Aug/train'
    #json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json_Aug/train'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat_Aug/'
    #test_csv = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/Gleason_masks_test_pathologist1.csv'
    csv_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/5Crops.csv'
    epoches = 1

    #image_folder = '/data/hdd1/by/TCGA_Prostate/Image_Test'
    #json_folder = '/data/hdd1/by/TCGA_Prostate/Json_Test'
    #save_path = '/data/hdd1/by/TCGA_Prostate/Feat_Test_No_Random/0'

    image_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/5Crops_Aug/'
    #json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json_Aug/train'
    json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/5Crops_Aug/'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat_Aug/Before_FC'
    save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat/5Crops_Aug_Color'

    datasets = []
    num_feats = []
    rel_pathes = []
    batch_size = 3000 * 3
    extractor = ExtractorResNet50(resnet50_path)
    #dataset = Prosate(image_folder, json_folder)
    #dataset = Prostate5Crops(image_folder, json_folder, csv_path)
    dataset = Prostate5CropsAug(image_folder, json_folder)
    #dataset = ProstateTest(image_folder, json_folder, test_csv)
    print(len(dataset), 'dataset length')
    #data_writer = FeatLMDBWriter
    data_writer = TorchWriter
    #start = time.time()
    count = 0
    for rel_path, image, bboxes, coords in dataset:
        count += 1
        print(count)
        patch_dataset = Patches(image, coords, bboxes, 64)
        #if len(bboxes) == 0:
        #    print(rel_path)
        #    print(len(patch_dataset))
        #for p in patch_dataset:
        #    pass

        #image = patch_dataset.image
        #image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
        #cv2.imwrite('im.jpg', image)

        #import sys; sys.exit()
        datasets.append(patch_dataset)
        #print(num_feat, len(patch_dataset))
        num_feats.append(len(patch_dataset))
        rel_pathes.append(rel_path)


        #count += sum(num_feats)
        if count % 50 == 0:

            datasets = torch.utils.data.ConcatDataset(datasets)
            data_loader = DataLoader(datasets, num_workers=4, batch_size=batch_size, shuffle=False)

            for epoch in range(epoches):
                save_path_epoch = os.path.join(save_path, str(epoch))
                os.makedirs(save_path_epoch, exist_ok=True)
                print('saving data to {}.....'.format(save_path_epoch))
                writer = data_writer(save_path_epoch)
                generate_features(data_loader, num_feats, rel_pathes, writer, extractor)
                #import sys;sys.exit()
            datasets = []
            num_feats = []
            rel_pathes = []

    datasets = torch.utils.data.ConcatDataset(datasets)
    data_loader = DataLoader(datasets, num_workers=4, batch_size=3000, shuffle=False)
    for epoch in range(epoches):
        save_path_epoch = os.path.join(save_path, str(epoch))
        os.makedirs(save_path_epoch, exist_ok=True)
        print('saving data to {}.....'.format(save_path_epoch))
        writer = data_writer(save_path_epoch)
        generate_features(data_loader, num_feats, rel_pathes, writer, extractor)

    print(count)
