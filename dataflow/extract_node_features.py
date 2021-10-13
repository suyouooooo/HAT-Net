import json
import os
from pathlib import Path
from functools import partial
import sys
import csv
import random
import pickle
import glob
import string
import time
import cv2

sys.path.append(os.getcwd())
from skimage.filters.rank import entropy as Entropy
from skimage.morphology import disk
from common.nuc_feature import nuc_stats_new,nuc_glcm_stats_new


#sys.path.append('../')

import lmdb
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import adaptive_avg_pool3d, adaptive_avg_pool2d
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

class PatchesCombined:
    def __init__(self, image, coords, bboxes, patch_size):
        """
            bboxes :bbox [min_y, min_x, max_y, max_x]
            coords: (row, col)
        """

        from model.cpc import get_transforms
        self.image = image
        self.bboxes = bboxes
        self.patch_size = patch_size
        self.coords = coords
        #from model.cpc import network
        self.transforms = get_transforms()
        #mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr
        #std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        #mean = [0.73646324, 0.56556627, 0.70180897] # Expanuke bgr
        #self.std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        #std = [0.18869222, 0.21968669, 0.17277594] # Expanuke bgr


    #def entory(self, gray_image, d):
    #    return Entropy(gray_image, d)

    def extract_func(self, image, mask):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask[mask > 0] = 1
        binary_mask = mask

        entropy = Entropy(gray_image, disk(3))
        mean_im_out, diff, var_im, skew_im = nuc_stats_new(binary_mask, gray_image)
        glcm_feat = nuc_glcm_stats_new(binary_mask, gray_image) # just breakline for better code
        glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_ASM = glcm_feat
        mean_ent = cv2.mean(entropy, mask=gray_image)[0]
        info = cv2.findContours(binary_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = info[0][0]
        num_vertices = len(cnt)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            hull_area += 1
        solidity = float(area)/hull_area
        if num_vertices > 4:
            centre, axes, orientation = cv2.fitEllipse(cnt)
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
        else:
            orientation = 0
            majoraxis_length = 1
            minoraxis_length = 1
        perimeter = cv2.arcLength(cnt, True)
        eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)


        nuc_feats = []
        #nuc_feats.append(mean_im_out)
        nuc_feats.append(diff)
        nuc_feats.append(var_im)
        nuc_feats.append(skew_im)
        nuc_feats.append(mean_ent)
        nuc_feats.append(glcm_dissimilarity)
        nuc_feats.append(glcm_homogeneity)
        nuc_feats.append(glcm_energy)
        nuc_feats.append(glcm_ASM)
        nuc_feats.append(eccentricity)
        nuc_feats.append(area)
        nuc_feats.append(majoraxis_length)
        #nuc_feats.append(minoraxis_length) #
        nuc_feats.append(perimeter)
        #nuc_feats.append(solidity)
        #nuc_feats.append(orientation) #
        #nuc_feats = np.array(nuc_feats)

        #print(nuc_feats.shape)

        return nuc_feats

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, idx):
        bbox = self.bboxes[idx]
        #print(bbox, idx)
        #bbox = self.pad_patch(*bbox, self.patch_size)
        patch = self.image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
        image = patch[:, :, :3]
        mask = patch[:, :, 3:]
        coord = self.coords[idx]
        hand_features = self.extract_func(image, mask[:, :, 0])
        #hand_features = self.extract_func(image, mask)
        #print(hand_features.shape, 11)
        #print(coord.shape, 22)
        #hand_features = torch.cat([self.extract_func(image, mask[:, :, 0])])
        coord.extend(hand_features)
        hand_features_and_coord = np.array(coord)
        #print(hand_features_and_coord.shape, hand_features_and_coord.sum())

        bbox = self.pad_patch(*bbox, self.patch_size)
        #print(bbox[2] - bbox[0], bbox[3] - bbox[1], self.patch_size, bbox)
        #print(image.shape, 1111111111)
        patch = self.image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1][:, :, :3]
        top = max(int((patch.shape[0] - 64) / 2), 0)
        bottom = max(0, top - patch.shape[0])
        right = max(int((patch.shape[1] - 64) / 2), 0)
        left = max(0, right - patch.shape[1])
        #print(top, bottom, left, right)
        #print('before pad', patch.shape)
        patch = cv2.copyMakeBorder(patch, top, bottom, left, right, cv2.BORDER_REFLECT)
        #print('after pad', patch.shape)
        #print(image.shape)
        patch = self.transforms(patch)
        #print(patch.shape)

        #print(patch.shape)
        return patch, hand_features_and_coord
        #patch = self.transforms(patch)
        #return patch, np.array(self.coords[idx])

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


class PatchesHandCrafted:
    def __init__(self, image, coords, bboxes, patch_size):
        """
            bboxes :bbox [min_y, min_x, max_y, max_x]
            coords: (row, col)
        """

        self.image = image
        self.bboxes = bboxes
        self.patch_size = patch_size
        self.coords = coords
        #mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr
        #std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        #mean = [0.73646324, 0.56556627, 0.70180897] # Expanuke bgr
        #self.std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        #std = [0.18869222, 0.21968669, 0.17277594] # Expanuke bgr


        #self.transforms = transforms.Compose([
        #    transforms.ToPILImage(),
        #    transforms.RandomChoice([
        #        transforms.RandomResizedCrop(patch_size, scale=(0.9, 1.1)),
        #        transforms.Resize((patch_size, patch_size))
        #    ]),
        #    transforms.RandomApply(torch.nn.ModuleList([
        #            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
        #    ]), p=0.3),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean, std),
        #])

    #def entory(self, gray_image, d):
    #    return Entropy(gray_image, d)

    def extract_func(self, image, mask):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask[mask > 0]= 1
        binary_mask = mask

        entropy = Entropy(gray_image, disk(3))
        mean_im_out, diff, var_im, skew_im = nuc_stats_new(binary_mask, gray_image)
        glcm_feat = nuc_glcm_stats_new(binary_mask, gray_image) # just breakline for better code
        glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_ASM = glcm_feat
        mean_ent = cv2.mean(entropy, mask=gray_image)[0]
        info = cv2.findContours(binary_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = info[0][0]
        num_vertices = len(cnt)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            hull_area += 1
        solidity = float(area)/hull_area
        if num_vertices > 4:
            centre, axes, orientation = cv2.fitEllipse(cnt)
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
        else:
            orientation = 0
            majoraxis_length = 1
            minoraxis_length = 1
        perimeter = cv2.arcLength(cnt, True)
        eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)


        nuc_feats = []
        nuc_feats.append(mean_im_out)
        nuc_feats.append(diff)
        nuc_feats.append(var_im)
        nuc_feats.append(skew_im)
        nuc_feats.append(mean_ent)
        nuc_feats.append(glcm_dissimilarity)
        nuc_feats.append(glcm_homogeneity)
        nuc_feats.append(glcm_energy)
        nuc_feats.append(glcm_ASM)
        nuc_feats.append(eccentricity)
        nuc_feats.append(area)
        nuc_feats.append(majoraxis_length) # nan
        nuc_feats.append(minoraxis_length)
        nuc_feats.append(perimeter)  # nan
        nuc_feats.append(solidity) # nan
        nuc_feats.append(orientation)
        nuc_feats = np.array(nuc_feats)

        #print(nuc_feats.shape)

        return nuc_feats

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, idx):
        bbox = self.bboxes[idx]
        #print(bbox, idx)
        #bbox = self.pad_patch(*bbox, self.patch_size)
        patch = self.image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
        image = patch[:, :, :3]
        mask = patch[:, :, 3:]
        features = self.extract_func(image, mask[:, :, 0])
        #patch = self.transforms(patch)
        return features, np.array(self.coords[idx])

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

class Patches:
    def __init__(self, image, coords, bboxes, patch_size):
        """
            bboxes :bbox [min_y, min_x, max_y, max_x]
            coords: (row, col)
        """

        self.image = image
        self.bboxes = bboxes
        self.patch_size = patch_size
        self.coords = coords
        #mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr consep
        #std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        #mean = [0.73646324, 0.56556627, 0.70180897] # Expanuke bgr
        #std = [0.18869222, 0.21968669, 0.17277594] # Expanuke bgr

        #self.std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        #self.std = [0.18869222, 0.21968669, 0.17277594] # Expanuke bgr
        mean = [0.72369437, 0.44910724, 0.68094617] # prostate bgr
        std = [0.17274064, 0.20472058, 0.20758244] # prosate bgr


        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomChoice([
                #transforms.RandomResizedCrop(patch_size, scale=(0.9, 1.1)),
                transforms.Resize((patch_size, patch_size))
            ]),
            #transforms.RandomApply(torch.nn.ModuleList([
            #        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
            #]), p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, idx):
        bbox = self.bboxes[idx]
        #print(bbox, idx)
        bbox = self.pad_patch(*bbox, self.patch_size)
        patch = self.image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
        patch = self.transforms(patch)
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

class ExtractorResNet50ImageNet:
    def __init__(self):
        self.net = network('resnet50',  5, True)
        self.net = self.net.cuda()
        print(torch.cuda.device_count())
        self.net = torch.nn.DataParallel(self.net)

    def __call__(self, images):
        with torch.no_grad():
            output = self.net(images.cuda())
            output = output.unsqueeze(0)
            output = adaptive_avg_pool3d(output, (16, 1, 1))
            output = output.squeeze()

        return output.cpu().numpy()

#### generating features
class ExtractorResNet50:
    def __init__(self, path):
        #self.net = network('resnet50',  5, False)
        self.net = network('resnet50', 6, False)
        self.net = self.net.cuda()
        self.net = torch.nn.DataParallel(self.net)
        print('loading weight file {}...'.format(path))
        self.net.load_state_dict(torch.load(path))
        print('Done.')

    def __call__(self, images):
        with torch.no_grad():
            output = self.net(images.cuda())
            output = output.unsqueeze(0)
            output = adaptive_avg_pool3d(output, (512, 1, 1))
            output = output.squeeze()

        return output.cpu().numpy()
        #return output.cpu()

#def extractor_resnet(images, mask=None):
    #net = network('resnet50',  4, False)
    #net = net.load_state_dict(torch.load(res50_path))
    #node_features = net(images)
    #return node_features

class ExtractNodeFeat:
    def __init__(self, patch_size, extractor):
        self.dataset = []
        self.extractor = extractor
        self.length = []
        self.patch_size = patch_size
        self.path = []
        self.mask = []

    def update(self, image, path, bboxes, coords, mask):
        assert len(bboxes) == len(coords)
        dataset = Patches(image, coords, bboxes, self.patch_size)
        self.dataset.append(dataset)
        self.length.append(len(bboxes))
        self.path.append(path)
        self.mask.append(mask)

    def run(self, batch_size):
        dataset = torch.utils.data.ConcatDataset(self.dataset)
        data_loader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=False, collate_fn=object_list)

        node_features = []
        node_coords = []
        #with torch.no_grad():
        with torch.no_grad():
            for images, coords in data_loader:
                images = torch.stack(images)
                print(images.shape)
                output = self.extractor(images)
                output = output.unsqueeze(0)
                output = adaptive_avg_pool3d(output, (16, 1, 1))
                output = output.squeeze()
                node_features.append(output.cpu().numpy())
                node_coords.extend(coords)

        if node_features:
            node_features = np.vstack(node_features)
        else:
            node_features = np.array(node_features)

        if node_coords:
            node_coords = np.vstack(node_coords)
        else:
            node_coords = np.array(node_coords)

        assert len(node_coords) == len(node_features)

        feats = []
        coords = []
        prev = 0
        for l in self.length:
            feats[prev:prev + l]
            coords[prev:prev + l]
            prev += l

        assert prev == sum(self.length)

        self.dataset = []
        self.length = []
        path = self.path
        self.path = []
        mask = self.mask
        self.mask = []

        assert len(feats) == len(coords) == len(path) == len(mask)

        return feats, coords, path, mask




def extract_node_features_resnet(image, bboxes, coords, patch_size, batch_size, extractor):
    import time
    start = time.time()
    dataset = Patches(image, coords, bboxes, patch_size)
    data_loader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=False)

    node_features = []
    node_coords = []
    #with torch.no_grad():
    with torch.no_grad():
        for images, coords in data_loader:
        #coords = []
        #for image, coord in dataset:
            #node_features = net(images)
            output = extractor(images)
            output = output.unsqueeze(0)
            output = adaptive_avg_pool3d(output, (16, 1, 1))
            output = output.squeeze()

            node_features.append(output.cpu().numpy())

            node_coords.append(coords.numpy())



        if node_features:
            node_features = np.vstack(node_features)
        else:
            node_features = np.array(node_features)

        if node_coords:
            node_coords = np.vstack(node_coords)
        else:
            node_coords = np.array(node_coords)

        print(len(node_coords))
        assert len(node_coords) == len(node_features)

    finish = time.time()
    print(finish - start)
    return node_features, node_coords


def extract_node_features_hand_crafted(image, bboxes, patch_size, batch_size, extractor):
    pass

class ImageNumpy:
    def __init__(self, image_folder, label_folder, return_mask=False):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_names = []
        self.return_mask = return_mask
        search_path = os.path.join(image_folder, '**', '*.png')
        for image_name in glob.iglob(search_path, recursive=True):
            if 'mask' in image_name:
                continue
            self.image_names.append(image_name)

        print(len(self.image_names))
    def __len__(self):
        return len(self.image_names)

    def image2mask_fp(self, image_path):
        rel_image_path = Path(image_path).relative_to(self.image_folder)
        mask_path = os.path.join(self.label_folder, str(rel_image_path).replace('.png', '.npy'))
        return mask_path

    def __getitem__(self, idx):
        image_path = self.image_names[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        bboxes = []
        coords = []

        mask_path = self.image2mask_fp(image_path)
        mask = np.load(mask_path)
        props = regionprops(mask)
        for prop in props:
            bboxes.append([int(c) for c in prop.bbox])
            coords.append([int(c) for c in prop.centroid])

        rel_image_path = str(Path(image_path).relative_to(self.image_folder))
        if self.return_mask:
            #return image, rel_image_path, bboxes, coords, mask
            return rel_image_path, np.concatenate((image, mask), axis=2), bboxes, coords
            #return np.rel_image_path, image, bboxes, coords, mask
        else:
            return rel_image_path, image, bboxes, coords

class Prosate5CropsAugCPC:
    def __init__(self, image_folder, json_folder, return_mask=True):
        self.image2path = {}
        self.json2path = {}
        #search_path = os.path.join(image_folder, '**', '*.jpg')
        search_path = os.path.join(image_folder, '**', '*.png')
        for image_fp in glob.iglob(search_path, recursive=True):
            image_name = os.path.basename(image_fp)
            self.image2path[image_name] = image_fp

        search_path = os.path.join(json_folder, '**', '*.json')
        for json_fp in glob.iglob(search_path, recursive=True):
            json_name = os.path.basename(json_fp)
            self.json2path[json_name] = json_fp

        self.image_names = list(self.image2path.keys())

        self.return_mask = return_mask
        self.scale = 1

        #self.image2label = self.read_csv(csv_file)

        #self.image_names = []
        #for key in self.image2label.keys():
        #    self.image_names.append(key)


#    def read_csv(self, csv_file):
#        res = {}
#        with open(csv_file, newline='') as csvfile:
#            reader = csv.DictReader(csvfile)
#            for row in reader:
#                gs = int(row['gleason score'])
#                if gs <= 6:
#                   res[row['image name']] = 1
#                else:
#                   res[row['image name']] = 2
#        return res

    def __len__(self):
        return len(self.image2path.keys())

    def read_file(self, json_path):
        res = []
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            #print(res.keys())
            #print(res['nuc']['2175'])
            #print(len(res['nuc'].keys()))
            for k, v in json_data['nuc'].items():
                #print(type(v))
                #print(v.keys())
                res.append(v)
        return res

    def format_labels(self, labels):
        coords = []
        bboxes = []
        if self.return_mask:
            mask = np.zeros((self.image_size * self.scale, self.image_size * self.scale), dtype='uint8')

        type_probs = []
        for node in labels:
            cen = node['centroid']
            cen = [int(c) for c in cen]
            cen = cen[::-1]
            coords.append(cen)
            #image = cv2.circle(image, tuple(cen), 3, (0, 200, 0), cv2.FILLED, 1)
            # node['bbox'] : [[rmin, cmin], [rmax, cmax]]
            bbox = node['bbox']
            #print(node['type_prob'])
            # bbox : [min_y, min_x, max_y, max_x]
            #bbox = [b // 2 for b in sum(bbox, [])]
            bbox = [b for b in sum(bbox, [])]
            bboxes.append(bbox)

            #contour = node['contour']
            #node['contour'] = [[c1 // 2, c2 // 2] for [c1, c2] in contour]
            #type_probs.append(node['type_prob'])

            if self.return_mask:
                #cv2.imwrite('heihei_del1.png', image)
                cv2.drawContours(mask, [np.array(node['contour'])], -1, 255, -1)
                #cv2.imwrite('heihei_del.png', image)
                #cv2.imwrite()

        if self.return_mask:
            #return bboxes, coords, mask, type_probs
            return bboxes, coords, mask
        else:
            #return bboxes, coords, None, type_probs
            return bboxes, coords, None

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_fp = self.image2path[image_name]
        image = cv2.imread(image_fp, -1)
        self.image_size = image.shape[0]

        #json_fp = self.json2path[image_name.replace('.jpg', '.json')]
        json_fp = self.json2path[image_name.replace('.png', '.json')]
        #print(image_fp, json_fp)
        #bboxes, coords = self.read_json(json_fp)
        labels = self.read_file(json_fp)
        bboxes, coords, mask, type_probs = self.format_labels(labels)

        image_name = os.path.basename(image_fp)
        #label = self.image2label[image_name]
        #print(label)
        #image_name = image_name.replace('.', '_grade_{}.'.format(label))
        #print(image_name)

        if self.return_mask:
            #mask = np.unsqueeze(mask, axis=-1)
            mask = np.expand_dims(mask, axis=-1)
            #print(mask.shape, image.shape)
            #return os.path.basename(image_fp), np.concatenate((image, mask), axis=2), bboxes, coords
            return image_name, np.concatenate((image, mask), axis=2), bboxes, coords
            #return rel_image_path, np.concatenate((image, mask), axis=2), bboxes, coords

        #return image, rel_image_path, bboxes, coords, mask
        #return image, rel_image_path, bboxes, coords
        #return rel_image_path, image, bboxes, coords
        #image_name = os.path.basename(image_fp)
        #label = self.image2label[image_name]
        #image_name = image_name.replace('.', '_grade_{}.'.format(label))
        #print(image_name)
        #return os.path.basename(image_fp), image, bboxes, coord
        return image_name, image, bboxes, coords, type_probs

class Prosate5CropsCPC:
    def __init__(self, image_folder, json_folder, csv_file, return_mask=True):
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

        #self.image_names = list(self.image2path.keys())

        self.return_mask = return_mask
        self.scale = 1

        self.image2label = self.read_csv(csv_file)

        self.image_names = []
        for key in self.image2label.keys():
            self.image_names.append(key)


    def read_csv(self, csv_file):
        res = {}
        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                gs = int(row['gleason score'])
                if gs <= 6:
                   res[row['image name']] = 1
                else:
                   res[row['image name']] = 2
        return res

    def __len__(self):
        return len(self.image2path.keys())

    def read_file(self, json_path):
        res = []
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            #print(res.keys())
            #print(res['nuc']['2175'])
            #print(len(res['nuc'].keys()))
            for k, v in json_data['nuc'].items():
                #print(type(v))
                #print(v.keys())
                res.append(v)
        return res

    def format_labels(self, labels):
        coords = []
        bboxes = []
        if self.return_mask:
            mask = np.zeros((self.image_size * self.scale, self.image_size * self.scale), dtype='uint8')

        for node in labels:
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

            #contour = node['contour']
            #node['contour'] = [[c1 // 2, c2 // 2] for [c1, c2] in contour]

            if self.return_mask:
                #cv2.imwrite('heihei_del1.png', image)
                cv2.drawContours(mask, [np.array(node['contour'])], -1, 255, -1)
                #cv2.imwrite('heihei_del.png', image)
                #cv2.imwrite()

        if self.return_mask:
            return bboxes, coords, mask
        else:
            return bboxes, coords, None

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_fp = self.image2path[image_name]
        image = cv2.imread(image_fp, -1)
        self.image_size = image.shape[0]

        json_fp = self.json2path[image_name.replace('.jpg', '.json')]
        #print(image_fp, json_fp)
        #bboxes, coords = self.read_json(json_fp)
        labels = self.read_file(json_fp)
        bboxes, coords, mask = self.format_labels(labels)

        image_name = os.path.basename(image_fp)
        label = self.image2label[image_name]
        #print(label)
        image_name = image_name.replace('.', '_grade_{}.'.format(label))
        #print(image_name)

        if self.return_mask:
            #mask = np.unsqueeze(mask, axis=-1)
            mask = np.expand_dims(mask, axis=-1)
            #print(mask.shape, image.shape)
            #return os.path.basename(image_fp), np.concatenate((image, mask), axis=2), bboxes, coords
            return image_name, np.concatenate((image, mask), axis=2), bboxes, coords
            #return rel_image_path, np.concatenate((image, mask), axis=2), bboxes, coords

        #return image, rel_image_path, bboxes, coords, mask
        #return image, rel_image_path, bboxes, coords
        #return rel_image_path, image, bboxes, coords
        #image_name = os.path.basename(image_fp)
        #label = self.image2label[image_name]
        #image_name = image_name.replace('.', '_grade_{}.'.format(label))
        #print(image_name)
        #return os.path.basename(image_fp), image, bboxes, coord
        return image_name, image, bboxes, coord

class ImageJson:
    def __init__(self, image_folder, label_folder, image_size, return_mask=False, scale=2):
        self.image_folder = image_folder
        self.image_size = image_size
        self.label_folder = label_folder
        json_lists = JsonFolder(label_folder)
        self.json_dataset = JsonDataset(json_lists, image_size)
        image_list = ImageFolder(image_folder)
        self.image_dataset = ImageDataset(image_list, image_size)
        self.image_lists = self.image_dataset.file_path_lists
        self.return_mask = return_mask
        self.scale = 2
        #self.json_lists = self.json_dataset.file_path_lists
        #self.transform = transform

    def image_prefixes(self):
        return self.image_dataset.file_prefixes

    def image_lists_by_prefix(self, prefix):
        return self.image_dataset.file_grids[prefix]

    def get_res_by_image_path(self, image_path):
        image_path = Path(image_path)
        rel_image_path = str(image_path.relative_to(self.image_folder))
        image = self.image_dataset.get_file_by_path(str(image_path))
        image = cv2.resize(image, (0,0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        json_path = self.image2json_fp(image_path)
        labels = self.json_dataset.get_file_by_path(json_path)
        #print(type(labels))
        bboxes, coords, mask = self.format_labels(labels)
        return rel_image_path, image, bboxes, coords, mask


    def __getitem__(self, idx):
        image_path = Path(self.image_lists[idx])
        rel_image_path = str(image_path.relative_to(self.image_folder))
        image = self.image_dataset.get_file_by_path(str(image_path))
        image = cv2.resize(image, (0,0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        json_path = self.image2json_fp(image_path)
        labels = self.json_dataset.get_file_by_path(json_path)
        #print(type(labels))
        bboxes, coords, mask = self.format_labels(labels)

        if self.return_mask:
            #mask = np.unsqueeze(mask, axis=-1)
            mask = np.expand_dims(mask, axis=-1)
            #print(mask.shape, image.shape)
            return rel_image_path, np.concatenate((image, mask), axis=2), bboxes, coords

        #return image, rel_image_path, bboxes, coords, mask
        #return image, rel_image_path, bboxes, coords
        return rel_image_path, image, bboxes, coords

    def __len__(self):
        assert len(self.image_dataset) == len(self.json_dataset)
        return len(self.image_dataset)

    def image2json_fp(self, image_path):
        base_name = os.path.basename(image_path)
        json_file = base_name.split('.')[0] + '.json'
        return os.path.join(self.label_folder, json_file)

    def format_labels(self, labels):
        coords = []
        bboxes = []
        if self.return_mask:
            mask = np.zeros((self.image_size * self.scale, self.image_size * self.scale), dtype='uint8')

        #print(labels.keys())
        for node in labels:
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

            #contour = node['contour']
            #node['contour'] = [[c1 // 2, c2 // 2] for [c1, c2] in contour]

            if self.return_mask:
                #cv2.imwrite('heihei_del1.png', image)
                cv2.drawContours(mask, [np.array(node['contour'])], -1, 255, -1)
                #cv2.imwrite('heihei_del.png', image)
                #cv2.imwrite()

        if self.return_mask:
            return bboxes, coords, mask
        else:
            return bboxes, coords, None

class LMDBWriter:
    def __init__(self, save_path):
        self.datasets = {}
        self.save_path = save_path

    def __call__(self, key, value, printable=False):
        """key: relative path to image"""
        assert type(key) == str
        assert type(value) == bytes

        #image_name = os.path.basename(key)
        dir_name = os.path.dirname(key)

        if dir_name not in self.datasets:
            map_size = 10 << 40
            lmdb_path = os.path.join(self.save_path, dir_name)
            os.makedirs(lmdb_path, exist_ok=True)
            env = lmdb.open(lmdb_path, map_size=map_size)
            self.datasets[dir_name] = env

        env = self.datasets[dir_name]
        with env.begin(write=True) as txn:
            if printable:
                print('writing files to {}...'.format(
                    os.path.join(self.save_path, key)))
            txn.put(key.encode(), value)

class LMDBReader:
    def __init__(self, path, transform):
        self.env = lmdb.open(path, map_size=1099511627776, readonly=True, lock=False)
        #with self.env.begin(write=False) as txn:
        #    self.image_names= [key.decode() for key in txn.cursor().iternext(keys=True, values=False)]

        cache_file = '_cache_' + ''.join(c for c in path if c in string.ascii_letters)
        cache_path = os.path.join(path, cache_file)
        if os.path.isfile(cache_path):
            self.image_names = pickle.load(open(cache_path, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.image_names = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.image_names, open(cache_path, "wb"))

        self.transform = transform

    @classmethod
    def init_dataset(cls, save_path, transform):
        datasets = []
        for lmdb_fp in glob.iglob(os.path.join(save_path, '**', 'data.mdb'), recursive=True):
            #print(lmdb_fp)
            datasets.append(cls(os.path.dirname(lmdb_fp), transform=transform))

        return torch.utils.data.ConcatDataset(datasets)

    def __len__(self):
        return len(self.image_names)

    def knn(self, data):
        edge_index = radius_graph(data.pos, 100, None, True, 8)
        data.edge_index = edge_index

        return data

    def to_data(self, image_name, data):
        feat = data['feat']
        coord = data['coord']
        feat = np.concatenate((feat, coord), axis=-1)
        coord = torch.from_numpy(coord).to(torch.float)
        feat = torch.from_numpy(feat).to(torch.float)

        if '1_normal' in image_name:
            label = 0
        elif '2_low_grade' in image_name:
            label = 1
        else:
            label = 2

        y = torch.tensor([label], dtype=torch.long)
        data = Data(x=feat, pos=coord, y=y)

        return data
    #def __getitem__(self, idx):
    #    NotImplementedError
    def __getitem__(self, idx):

        image_name = self.image_names[idx]
        with self.env.begin(write=False) as txn:
            data = txn.get(image_name)
            data = pickle.loads(data)

        data = self.to_data(image_name.decode(), data)
        if self.transform:
            data = self.transform(data)

        data = self.knn(data)

        return image_name.decode(), data


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

def _avg_pool_x(cluster, x, size=None):
    return scatter(x, cluster, dim=0, dim_size=size, reduce='mean')

def avg_pooling(data):
    cluster = grid_cluster(data.pos, torch.Tensor([64, 64]))
    cluster, perm = consecutive_cluster(cluster)
    x = None if data.x is None else _avg_pool_x(cluster, data.x)
    pos = None if data.pos is None else pool_pos(cluster, data.pos)
    data.x = x
    data.pos = pos

    return data

def _add_pool_x(cluster, x, size=None):
    return scatter(x, cluster, dim=0, dim_size=size, reduce='add')

def add_pooling(data):
    cluster = grid_cluster(data.pos, torch.Tensor([64, 64]))
    cluster, perm = consecutive_cluster(cluster)
    x = None if data.x is None else _add_pool_x(cluster, data.x)
    pos = None if data.pos is None else pool_pos(cluster, data.pos)
    data.x = x
    data.pos = pos

    return data

def object_list(batch):
    #pathes = []
    #data = []
    #for p, d in batch:
    #    pathes.append(p)
    #    data.append(d)

    #print(len(batch))
    #res = [b for b in batch]
    #print(len(res))
    #return pathes, data
    return [b for b in zip(*batch)]

#def object_list(batch):

#def gen_training_data(image_folder, label_folder, extract_func, feature_type, image_size, save_path):
def gen_training_data(conf):
    """type: hand-crafted, resnet50"""

    dataset = conf.dataset
    #if feature_type == 'hand':
    #    extractor = extract_node_features_hand_crafted
    #    dataset = None
    #elif feature_type == 'res50':
    #    extractor = extract_node_features_resnet
    #    dataset = ImageJson(image_folder, label_folder, image_size)

    #####processed data

    #image, rel_image_path, bboxes, coords, mask = random.choice(dataset)
    data_loader = DataLoader(dataset, num_workers=2, batch_size=16, shuffle=False, collate_fn=object_list)
    print('extracting node features......')
    count = 0
    for image, rel_image_path, bboxes, coords, mask in data_loader:

        for img, p, box, coord, m in zip(image, rel_image_path, bboxes, coords, mask):

        #    count += 1
        #    if count % 10 != 0:
        #        print(conf.node_feature_coords)
        #        conf.node_feature_coords.update(img, p, box, coord, m)
        #        continue


        #print(bboxes)
    #for coord in coords:
    #        image = cv2.circle(image, tuple(coord[::-1]), 3, (0, 200, 0), cv2.FILLED, 3)

    #image = cv2.resize(image, (0, 0), fx=0.125, fy=0.125)
    #cv2.imwrite('/home/baiyu/HGIN/heihei_del11.png', image)
    #sys.exit()
    #for bbox in bboxes:
    #    image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (255, 0, 0), thickness=1)

    ##image = cv2.drawContours(image, [np.array(node['contour']) // 2 ], -1, (0, 0, 0), -1)

    #image[mask != 0] = 0
    #cv2.imwrite('heihei_del11.png', image)
    #sys.exit()
    ############ saving node_features and node_coords
            node_features, node_coords = conf.node_feature_coords(img, box, coord)
            #node_features, node_coords, path, mask = conf.node_feature_coords.run(3000)

            #for node_feat, node_coord, p, m in zip(node_features, node_coords, path, mask):

            res = {
                'feat': node_features,
                'coord' : node_coords,
                #'mask' : mask
                'mask' : m
            }

            val = pickle.dumps(res)
            count += 1
            print('[{}/{}]'.format(count, len(dataset)))
            conf.writer(str(p), val, conf.print)

    ######################## read features
    print('generating cell graph....')
    dataset = conf.reader(conf.save_path)
    save_path = conf.training_data_path
    #from torch_geometric.data import DataLoader

    data_loader = DataLoader(dataset, num_workers=4, batch_size=128, shuffle=False, collate_fn=object_list)

    for epoch in range(conf.epoches):
        for path, data in data_loader:

            for p, d in zip(path, data):
                os.makedirs(os.path.join(save_path, str(epoch)), exist_ok=True)
                fp = os.path.join(save_path, str(epoch), p.split('.')[0] + '.pt')
                print(fp)
                #torch.save(d, fp)



    #for image_name, val in dataset:
        #print(image_name)
        #image_name.
        #for k, v in val.items():
            #if k == 'mask':
                #continue
            #print(k, v.shape)




    #for coord in node_coords:
    #    image = cv2.circle(image, tuple(coord[::-1]), 3, (0, 200, 0), cv2.FILLED, 1)

    #cv2.imwrite('heihei_del11.png', image)

class Res50BaseConfig:
    def __init__(self):
        self.epoches = 1
        self.image_size = 224
        self.print = True

        self.image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC/'
        self.label_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_su/proto/mask/CRC'
        self.save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data/crc/res50_1792_avg_knn/proto/'
        self.pool = 'avg'
        transform = avg_pooling
        patch_size = 64
        #self.pool = 'add':
        #transform = add_pooling

        self.method = 'knn'
        self.seg = 'cia'
        folder_name = 'fix_{}_{}_{}'.format(self.pool, self.seg, self.method)
        self.training_data_path = os.path.join('/home/baiyu/training_data/CRC', folder_name)
        self.return_mask = True
        #res50_path = '/home/baiyu/HGIN/checkpoint/191-best.pth'
        #res50_path = 'checkpoint/resnet50/Tuesday_01_June_2021_19h_43m_05s/191-best.pth' # colon
        res50_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ResNet50/97-best.pth'

        #mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr
        #std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        #self.valid_transforms = transforms.Compose([
        #    transforms.ToPILImage(),
        #    transforms.Resize((patch_size, patch_size)),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean, std),
        #])
        #self.dataset = ImageNumpy(self.image_path, self.label_path, self.image_size, self.return_mask, transform=valid_transforms)
        self.extract_func = ExtractorResNet50(res50_path)
        #self.writer = LMDBWriter(self.save_path)
        #self.reader = partial(LMDBReader.init_dataset, transform=transform)
        #self.node_feature_coords = ExtractNodeFeat(patch_size=patch_size,
        #                                                    extractor=self.extract_func)
        #extract_node_feat
        self.node_feature_coords = partial(extract_node_features_resnet,
                                        batch_size=3000,
                                        patch_size=patch_size,
                                        extractor=self.extract_func,
                                        )

class Res50NumpyMaskConfig(Res50BaseConfig):
    def __init__(self):
        super().__init__()

        self.image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC/'
        self.label_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_su/proto/mask/CRC'
        self.save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_lmdb/crc/res50_1792_avg_knn/proto/'
        self.pool = 'avg'
        transform = avg_pooling
        #self.pool = 'add':
        #transform = add_pooling

        self.seg = 'cia'
        folder_name = 'fix_{}_{}_{}'.format(self.pool, self.seg, self.method)
        self.training_data_path = os.path.join('/home/baiyu/training_data/CRC', folder_name)
        self.return_mask = True
        self.dataset = ImageNumpy(self.image_path, self.label_path, self.image_size, self.return_mask)
        self.writer = LMDBWriter(self.save_path)
        self.reader = partial(LMDBReader.init_dataset, transform=transform)


class Res50JsonMaskConfig(Res50BaseConfig):
    def __init__(self):
        super().__init__()
        self.image_path = '/data/by/tmp/HGIN/test_can_be_del3'
        self.label_path = '/data/by/tmp/HGIN/test_can_be_del2/EXtended_CRC_Mask'
        #self.label_path = '/home/baiyu/EXtended_CRC_Mask'
        self.save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_lmdb/extended_crc/feat/'

        self.pool = 'avg'
        transform = avg_pooling
        #self.pool = 'add'
        #transform = add_pooling

        self.seg = 'hover'
        folder_name = 'fix_{}_{}_{}'.format(self.pool, self.seg, self.method)
        self.training_data_path = os.path.join('/home/baiyu/training_data/ExtendedCRC', folder_name)
        self.return_mask = False
        #self.dataset = ImageJson(self.image_path, self.label_path, self.image_size, self.return_mask)
        #self.dataset = ImageJson(self.image_path, self.label_path, self.image_size, self.return_mask)
        #self.reader = lmdb_concatenate(self.save_path, transform=transform)
        #self.writer = LMDBWriter(self.save_path)
        #self.reader = partial(LMDBReader.init_dataset, transform=transform)

class TorchWriter:
    def __init__(self, save_path):
        self.save_path = save_path

    def add_pair(self, pairs):
        for pair in pairs:
            name, val = pair
            name = name.split('.')[0] + '.pt'
            save_path = os.path.join(self.save_path, name)
            #if val['feat'].sum() == 0:
            #    print(name, val)
            #print(save_path)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(save_path)
            #print(val['feat'].shape)
            torch.save(val, save_path)


class FeatLMDBWriter:
    def __init__(self, save_path):
        map_size = 10 << 40
        self.env = lmdb.open(save_path, map_size=map_size)

    def add_pair(self, pairs):
        with self.env.begin(write=True) as txn:
            for pair in pairs:
                txn.put(*pair)

class ProsateTest:
    def __init__(self, image_folder, json_folder, csv_file):
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

        self.image_names = []
        self.labels = []
        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.image_names.append(row['image name'])
                self.labels.append(int(row['gleason score']) - 5)


        #self.image_names = list(self.image2path.keys())


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
        image_fp = self.image2path[image_name]
        image = cv2.imread(image_fp, -1)

        json_fp = self.json2path[image_name.replace('.jpg', '.json')]
        #print(image_fp, json_fp)
        bboxes, coords = self.read_json(json_fp)
        base_name = os.path.basename(image_fp)
        label = self.labels[idx]
        base_name = base_name.replace('.', '_grade_{}.'.format(label))

        return base_name, image, bboxes, coords

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

def generate_features(dataloader, num_feats, rel_pathes, writer, extractor):
    node_features = []
    node_coords = []
    #wih torch.no_grad():
    #torch.set_printoptions(sci_mode=False)

    #from collections import Counter
    #counter = Counter()
    #count = 0
    #total = 0
    for idx, (images, coords) in enumerate(data_loader):
        #print(images.shape)
        output = extractor(images)
        #print(output.shape)



        # combined
        #print(output.shape, coords.shape)
        #output = np.hstack([output, coords[:, 2:]])
        #coords = coords[:, :2]



        #print(output.shape, coords.shape, coords)
        #import sys; sys.exit()

        #ff = torch.nn.functional.softmax(output, dim=1)
        #ind = torch.argmax(ff, dim=1)
        ##print(ind, ff.max())
        #print(ind.shape)
        #cc = ff.max(dim=1)[0] > 0.5
        #total += len(ff)
        #count += cc.sum()
        #print(count / total, '3333')
        ##print(ind)
        ##for i in ind.tolist():
        #counter.update(ind.tolist())
        #print(counter)
        #print('---------------')
        #print(ff, ff.max())

        node_features.append(output)
        #node_coords.extend(coords)
        node_coords.append(coords)
        #print(output.shape, coords.shape)
        #print(len(node_features), 'fffff`')
        #print(len(node_coords), 'cccccc')
        #print(idx, len(data_loader))

    #sys.exit()
    #print(len(node_features))
    node_features = np.vstack(node_features)
    node_coords = np.vstack(node_coords)
    #print(node_features.shape, node_coords.shape, 111111111)
    assert len(node_features) == len(node_coords)
    #print(node_features.shape)
    #print(node_coords.shape)
    prev = 0
    #print('here...........................................')
    #print(len(rel_pathes), len(num_feats))
    #test_cum = 0
    lmdb_pair = []
    for rel_path, num_feat in zip(rel_pathes, num_feats):
        #print(rel_path, num_feat)

        sub_feat = node_features[prev : prev+num_feat]
        sub_coord = node_coords[prev : prev+num_feat]
        #print(sub_feat.shape, sub_coord.shape)
        val = {
            'feat' : torch.tensor(sub_feat),
            'coord' : torch.tensor(sub_coord)
        }
        #val = pickle.dumps(val)
        rel_path = rel_path.replace('.jpg', '.npy')
        #print(rel_path, sub_feat.shape, sub_coord.shape)
        prev += num_feat
        lmdb_pair.append([rel_path, val])

    #import sys; sys.exit()
    writer.add_pair(lmdb_pair)
    #print(count / (time.time() - start))

class ExtractorCPC:
    def __init__(self, weight_file):
        from model.cpc import network
        #print('loading weight file {}...'.format(path))
        self.net = network(weight_file)
        #path = ''
        #self.net.load_state_dict(torch.load(path))
        #print('Done.')
        self.net = self.net.cuda()
        self.net = torch.nn.DataParallel(self.net)

    def __call__(self, images):
        with torch.no_grad():
            output = self.net(images.cuda())
            output = output.unsqueeze(0)
            output = adaptive_avg_pool2d(output, (1, 1))
            output = output.squeeze()

        return output.cpu().numpy()



class ExtractHandCrafted:
    def __init__(self):
        pass

    def __call__(self, batch):
        return batch





if __name__ == '__main__':

    #dataset = ImageJson('/home/baiyu/Extended_CRC', '/home/baiyu/EXtended_CRC_Mask', 1792)

    #import sys;
    #dataset = ImageJson('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Images/', '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Json/EXtended_CRC_Mask/', 224, return_mask=True)
    #print(len(dataset))
    #import sys; sys.exit()
    #print(len(dataset))
    #print(dataset[2][0], dataset[2][1].shape, dataset[2][2], dataset[2][3])
    #print(dataset[2333][1].shape)
    #image = dataset[2][1]
    #import random

    #image = random.choice(dataset)[1]
    #img = image[:, :, :3]
    #mask = image[:, :, 3:]
    #img[mask[:, :, 0] != 0, :] = 0
    #print(np.unique(mask))
    ##img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    #print(img.shape)

    #cv2.imwrite('fff.jpg', img)


    #import sys; sys.exit()
    #save_path = ''
    #dataset = ImageJson('/home/baiyu/Extended_CRC', '/home/baiyu/EXtended_CRC_Mask', 1792)
    #image_path = '/home/baiyu/Extended_CRC'
    #json_path = '/home/baiyu/EXtended_CRC_Mask'
    #config = Res50JsonMaskConfig()
    #resnet50_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ResNet50/97-best.pth'
    resnet50_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ResNet50/81-best.pth'
    #image_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images'
    #json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat/'
    #image_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images_Aug/train'
    #json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json_Aug/train'
    #image_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images_Aug/test'
    #json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json_Aug/test'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat_Aug/'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat_Aug/train'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat_Aug/test'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat_Test'
    #csv_file = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/Gleason_masks_test_pathologist1.csv'
    #image_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/5Crops'
    #json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/5Crops'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat/5Crops_CPC'
    image_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/5Crops_Aug/'
    #json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/5Crops_Aug/'
    json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/5Crops_json_withtypes_Aug/'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat/5Crops_Aug_CPC/'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat/ImageNetPretrain'
    save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat/5Crops_Res50_withtype_Aug'
    csv_file = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/5Crops.csv'
    epoches = 1
    #image_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC'
    #json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/proto/mask/CRC/shaban-cia'

    image_folder = '/data/smb/syh/colon_dataset/CoNSeP/Train/Overlay/'
    json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CoNSeP/Json'


    datasets = []
    num_feats = []
    rel_pathes = []
    extractor = ExtractorResNet50(resnet50_path)
    #extractor = ExtractHandCrafted()
    #extractor = ExtractorCPC()
    #extractor = ExtractorResNet50ImageNet()
    #dataset = Prosate(image_folder, json_folder)
    #dataset = Prosate5CropsCPC(image_folder, json_folder, csv_file)
    dataset = Prosate5CropsAugCPC(image_folder, json_folder, return_mask=False)

    #import random
    #np.random.seed(2020)
    #image_name, image, bboxes, coords, type_probs = random.choice(dataset)
    #count = 0
    #for idx, bbox in enumerate(bboxes):
    #    assert bbox[0] < bbox[2]
    #    assert bbox[1] < bbox[3]
    #    if type_probs[idx] < 0.5:
    #        count += 1
    #        image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (0, 9, 100), 2)
    #    else:
    #        image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (0, 255, 0), 2)

    ##for idx, coord in enumerate(coords):
    ##    image = cv2.circle(image, tuple(coord[::-1]), 3, (0, 200, 0), cv2.FILLED, 1)

    #image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
    #print(count)
    #cv2.imwrite('fff.jpg', image)







    #import sys; sys.exit()
    #dataset = ImageNumpy(image_folder, json_folder)

    #path, image, bboxes, coords = dataset[333]
    #for c in coords:
    #    image = cv2.circle(image, tuple(c[::-1]), 3, (0, 200, 0), cv2.FILLED, 1)


    #image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
    #cv2.imwrite('fff.jpg', image)
    #print(image.shape)
    #import sys;sys.exit()
    ##print(image.shape)
    #mask = image[:, :, 3:]
    #image = image[:, :, :3]
    #image[mask[:, :, 0] != 0, :] = 0
    #image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
    #cv2.imwrite('fff.jpg', image)
    #import sys; sys.exit()
    #dataset = ProsateTest(image_folder, json_folder, csv_file)
    datawriter = TorchWriter
    count = 0
    PatchData = Patches
    #PatchData = PatchesHandCrafted
    #PatchData = PatchesCombined  # mask needed
    print(len(dataset))
    start = time.time()
    #count = 0
    for rel_path, image, bboxes, coords in dataset:
        #count += 1
        #if count != 1153:
            #continue

        #print(rel_path)
        #print(image.shape)
        #for c in coords:
        #    image = cv2.circle(image, tuple(c[::-1]), 3, (0, 200, 0), cv2.FILLED, 1)
        #for bbox in bboxes:
        #    if bbox[0] > bbox[2]:
        #        print(rel_path)

        #    if bbox[1] > bbox[3]:
        #        print(rel_path)



        #for bbox in bboxes:
        #    image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (255, 0, 0), 3)
        #image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
        #cv2.imwrite('fff.jpg', image)
        ## draw nuclei here

        #print(image.shape)
        #mask = image[:, :, 3:]
        #print(mask.shape)
        #image = image[:, :, :3]
        #image[mask[:, :, 0] != 0, :] = 0
        #image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
        #cv2.imwrite('fff.jpg', image)
        #print(rel_path)
        #import sys; sys.exit()
        patch_dataset = PatchData(image, coords, bboxes, 71)
        datasets.append(patch_dataset)
        num_feats.append(len(patch_dataset))
        rel_pathes.append(rel_path)


        count += 1
        if count % 100 == 0:
            finish = time.time()

            print('[{}/{}], speed:{}'.format(count, len(dataset), count / (finish - start)))
            print('dataset size:', len(datasets), sum(num_feats))

            datasets = torch.utils.data.ConcatDataset(datasets)
            data_loader = DataLoader(datasets, num_workers=4, batch_size=3000 * 3, shuffle=False)

            for epoch in range(epoches):
                save_path_epoch = os.path.join(save_path, str(epoch))
                os.makedirs(save_path_epoch, exist_ok=True)
                print('saving data to {}.....'.format(save_path_epoch))
                writer = datawriter(save_path_epoch)
                generate_features(data_loader, num_feats, rel_pathes, writer, extractor)
            datasets = []
            num_feats = []
            rel_pathes = []

    if not datasets:
        import sys; sys.exit()

    datasets = torch.utils.data.ConcatDataset(datasets)
    data_loader = DataLoader(datasets, num_workers=4, batch_size=3000 * 3, shuffle=False)
    for epoch in range(epoches):
        save_path_epoch = os.path.join(save_path, str(epoch))
        os.makedirs(save_path_epoch, exist_ok=True)
        print('saving data to {}.....'.format(save_path_epoch))
        writer = datawriter(save_path_epoch)
        generate_features(data_loader, num_feats, rel_pathes, writer, extractor)

    print(count)