import os
import math
import random
import json
import glob
from collections import namedtuple
from functools import partial
import csv
#import torch_geometric
#from torch_geometric.data import Data
import cv2
import numpy as np

import stich
from stich import JsonDataset, JsonFolder, ImageDataset, ImageFolder

Node = namedtuple('Node', ['centroid', 'bbox', 'contour'])
"""
node bbox: [xmin, ymin, xmax, ymax]
             cen: [x, y]
"""
#
#
#
#
#
##def vertical_flip(image, label)
#
#
#import torch
#
#data = torch.load('/data/smb/syh/PycharmProjects/CGC-Net/data_res50/proto/fix_avg_cia_knn/0/fold_1/H09-00622_A2H_E_1_1_grade_2_1793_5153.pt')
#
#from torch_geometric import transforms as T
#trans = T.RandomFlip(1)
#print(data.pos)
#data = trans(data)
#print(data.pos)

def rotate270(image_size, point):
    w, h = image_size
    point = rotate180(image_size, point)
    point = rotate90(image_size, point)
    return point

def rotate180(image_size, point):
    w, h = image_size
    point = rotate90(image_size, point)
    point = rotate90((h, w), point)
    return point

def rotate90(image_size, point):
    """
        rotate clockwise 90 degrees
        image_size (w, h)
        point: (x, y)
    """
    w, h = image_size
    #print(w, h)
    x, y = point
    new_y = x
    new_x = h - y

    return (new_x, new_y)
#def rotate(origin, point, angle):
#    """
#    Rotate a point counterclockwise by a given angle around a given origin.
#
#    The angle should be given in radians.
#    """
#    ox, oy = origin
#    px, py = point
#
#    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
#    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
#    return int(qx), int(qy)

def flip(origin, point, axis):
    """
        point: (x, y)
        flip vertically axis=0
        flip herizontally axis=1
    """
    or_x, or_y = origin
    x, y = point
    #print('hello', origin, point)
    if axis == 0:
        #new_y = 2 * or_y - y
        new_y = or_y - y
        assert new_y >= 0
        return (x, new_y)
    elif axis == 1:
        #new_x = 2 * or_x - x
        new_x = or_x - x
        assert new_x >= 0
        return (new_x, y)
    else:
        raise ValueError('axis should only be 1 or 0')

class RandomChoice:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, image_data, nodes, image_trans=True):
        #for trans in self.trans:
        trans = random.choice(self.trans)
        image_data, nodes = trans(image_data, nodes, image_trans)

        return image_data, nodes

def rotate90_transform(image_data, nodes, image_trans=True):
    """
        rotate clockwise 90 degrees
        point: (x, y)
    """
    w = image_data.shape[1]
    h = image_data.shape[0]
    res = []
    for node in nodes:
        cen = rotate90((w, h), node.centroid)
        if min(node.bbox) < 0:
            print(node.bbox)
        b1 = rotate90((w, h), node.bbox[:2])
        b2 = rotate90((w, h), node.bbox[2:])
        bbox = construct_bbox(b1, b2)
        cnts = trans_cnt((w, h), node.contour, rotate90)
        #b2 = rotate90(b2)
        res.append(Node(centroid=cen, bbox=bbox, contour=cnts))

    if image_trans:
        image_data = cv2.rotate(image_data, cv2.ROTATE_90_CLOCKWISE)

    return image_data, res

def rotate180_transform(image_data, nodes, image_trans=True):

    w = image_data.shape[1]
    h = image_data.shape[0]
    res = []

    for node in nodes:
        cen = rotate180((w, h), node.centroid)
        b1 = rotate180((w, h), node.bbox[:2])
        b2 = rotate180((w, h), node.bbox[2:])
        bbox = construct_bbox(b1, b2)
        cnts = trans_cnt((w, h), node.contour, rotate180)
        #b2 = rotate90(b2)
        #res.append(Node(centroid=cen, bbox=[*b1, *b2], contour=cnts))
        res.append(Node(centroid=cen, bbox=bbox, contour=cnts))
        #res.append(Node(centroid=cen, bbox=[*b1, *b2], contour=None))

    if image_trans:
        image_data = cv2.rotate(image_data, cv2.ROTATE_180)
    #image_data, nodes = rotate90_transform(image_data, nodes, image_trans=True)
    return image_data, res

def rotate270_transform(image_data, nodes, image_trans=True):

    w = image_data.shape[1]
    h = image_data.shape[0]

    res = []
    for node in nodes:
        cen = rotate270((w, h), node.centroid)
        #cen = rotate90(cen)
        #cen = rotate90(cen)
        b1 = rotate270((w, h), node.bbox[:2])
        #b1 = rotate90(b1)
        #b1 = rotate90(b1)
        b2 = rotate270((w, h), node.bbox[2:])
        bbox = construct_bbox(b1, b2)
        #b2 = rotate90(b2)
        cnts = trans_cnt((w, h), node.contour, rotate270)
        #b2 = rotate90(b2)
        res.append(Node(centroid=cen, bbox=bbox, contour=cnts))

    if image_trans:
        image_data = cv2.rotate(image_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image_data, res


def trans_cnt(origin, cnts, trans_fnc):
    res = []
    for cnt in cnts:
        cnt = trans_fnc(origin, cnt)
        res.append(cnt)

    return res

def construct_bbox(b1, b2):
    top_left_x =  min(b1[0], b2[0])
    top_left_y =  min(b1[1], b2[1])
    bot_right_x = max(b1[0], b2[0])
    bot_right_y = max(b1[1], b2[1])

    return (top_left_x, top_left_y, bot_right_x, bot_right_y)

def flip_transform(image_data, nodes, axis, image_trans=True):
    """
        nodes: list of nodes
        node bbox: [xmin, ymin, xmax, ymax]
             cen: [x, y]
    """

    y, x = image_data.shape[:2]
    #origin = int(x / 2),  int(y / 2)
    origin = (x, y)

    #for k, nodes in json_data['nuc'].items():
    res = []

    def construct_bbox(b1, b2):
        top_left_x =  min(b1[0], b2[0])
        top_left_y =  min(b1[1], b2[1])
        bot_right_x = max(b1[0], b2[0])
        bot_right_y = max(b1[1], b2[1])

        return (top_left_x, top_left_y, bot_right_x, bot_right_y)
    #               b2(x2, y2')
    #                               ====>
    # b1'(x1, y1')

    # b1 (x1, y1)
    #               b2(x2, y2)

    for node in nodes:
        cen = node.centroid  # (x, y)
        #print(cen, axis)
        cen = flip(origin, cen, axis=axis)
        #print(cen, origin)
        b1 = flip(origin, node.bbox[:2], axis=axis)
        b2 = flip(origin, node.bbox[2:], axis=axis)
        bbox = construct_bbox(b1, b2)
        if bbox[0] >= bbox[2]:
            print(bbox, b2, 'check', node.bbox)
        if bbox[1] >= bbox[3]:
            print(bbox, b2, 'check', node.bbox)
        if min(bbox) < 0 :
            print(bbox, 'check', node.bbox)
        #if min(b2) <0 :
            #print(b2, 'check', node.bbox)

        trans_fnc = partial(flip, axis=axis)
        cnts = trans_cnt(origin, node.contour, trans_fnc)
        res.append(Node(centroid=cen, bbox=bbox, contour=cnts))

    if image_trans:
        image_data = np.flip(image_data, axis=axis)

    return image_data, res

def test_rorate90():
    import numpy as np
    import cv2

    w = 4
    h = 7
    image = np.arange(w * h).reshape(h, w)
    #image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    image_r = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image_r = cv2.rotate(image_r, cv2.ROTATE_90_CLOCKWISE)
    # (20, 40)
    #origin = (10, 20)
    #print(image.shape)
    image_size = image.shape[:2][::-1]
    print(image_size)
    #point = (20, 40)
    #point = (0, 0)
    for x in range(image_size[0]):
        for y in range(image_size[1]):
            point = (x, y)
            rotated = rotate90(image_size, point)
            #image_size_r = image_r.shape[:3][::-1]
            rotated = rotate90(image_size[::-1], rotated)

            pos = np.where(image_r == image[y, x])
            pos = pos[::-1]
            pos = [p.item() for p in pos]

            print('before:{}, after:{}, should be:{}'.format(point, rotated, pos))

    print(image)
    print(image_r)

    image = cv2.rotate(image_r, cv2.cv2.ROTATE_90_CLOCKWISE)
    print(image)

def inverse_format(nodes, scale):
    res = []
    for node in nodes:
            cen = node['centroid'] # x, y
            cen = [int(c / scale)  for c in cen]
            cen = cen[::-1]

            bbox = node['bbox']
            # bbox : [min_y, min_x, max_y, max_x]
            #bbox = [b // 2 for b in sum(bbox, [])]
            bbox = [b for b in sum(bbox, [])] # y, x
            bbox = [int(b / scale) for b in bbox]

            #contour[:, 0] += bbox[0]
            #contour[:, 1] += bbox[1]
            contour += bbox[:2]
            res.append(Node(centroid=cen, bbox=bbox, contour=contour))

def formatnode(nodes):
    res = []
    for node in nodes:
        cen = node['centroid'] # x, y

        bbox = node['bbox']
        bbox = [b for b in sum(bbox, [])] # y, x
        cnt = node['contour']

        res.append(Node(centroid=cen, bbox=bbox, contour=cnt))

    return res

def inverse_formatnode(nodes):
    res = []
    for node in nodes:
        #cen = nodes.centeroid
        res.append(node._asdict())
        bbox = res[-1]['bbox']
        bbox = [bbox[:2], bbox[2:]]
        res[-1]['bbox'] = bbox
        res[-1]['type_prob'] = None
        res[-1]['type'] = None

    return res


def inverse_formatxy(nodes, scale):
    res = []
    for node in nodes:
        cen = node.centroid
        cen = [int(c * scale)  for c in cen]

        bbox = node.bbox
        bbox = [int(b * scale) for b in bbox]
        bbox[:2] = bbox[:2][::-1] # convert to x, y
        bbox[2:] = bbox[2:][::-1]

        contour = [[int(x * scale), int(y * scale)] for [x, y] in node.contour]
        #contour = node['contour']
        res.append(Node(centroid=cen, bbox=bbox, contour=contour))

    return res

def formatxy(nodes, scale):

    # all elements to (x, y)
    res = []
    for node in nodes:
            cen = node.centroid
            cen = [int(c / scale)  for c in cen]

            bbox = node.bbox
            bbox = [int(b / scale) for b in bbox]
            bbox[:2] = bbox[:2][::-1] # convert to x, y
            bbox[2:] = bbox[2:][::-1]

            contour = [[int(x / scale), int(y / scale)] for [x, y] in node.contour]
            #contour = node['contour']
            res.append(Node(centroid=cen, bbox=bbox, contour=contour))

    return res


def save_json(path, nodes, mag=None):
    new_dict = {}
    inst_info_dict = {}
    #print('node length:', len(nodes), type(nodes))
    for inst_id, node in enumerate(nodes):
        inst_info_dict[inst_id] = {  # inst_id should start at 1
                "bbox": node['bbox'],
                "centroid": node['centroid'],
                "contour": node['contour'],
                "type_prob": None,
                "type": None,
        }



    json_dict = {"mag": mag, "nuc": inst_info_dict}  # to sync the format protocol
    with open(path, "w") as handle:
        json.dump(json_dict, handle)
    return new_dict

def draw_nuclei(image, nodes):
    #print(json_label)
    print(image.shape, 'draw_nuclei', type(image))
    for node in nodes:
        #print(11, node)
        cen = node.centroid
        cen = [int(c)  for c in cen]
        image = cv2.circle(image, tuple(cen), 3, (0, 200, 0), cv2.FILLED, 5)

        bbox = node.bbox
        image = cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color=(255, 0, 0), thickness=4)
        image = cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color=(255, 0, 0), thickness=4)

    return image

class BaseWSI:
    def __init__(self, image_path, json_path):

#image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Images/'
        json_lists = JsonFolder(json_path)
        image_lists = ImageFolder(image_path)

        self.image_dataset = ImageDataset(image_lists, 224)
        self.json_dataset = JsonDataset(json_lists, 224)
        self.image_prefixes = self.image_dataset.file_prefixes
        #self.transforms = transforms

        print(len(self.json_dataset))
        print(len(self.image_dataset))

        print(len(self.image_prefixes))

    def __len__(self):
        return len(self.image_prefixes)

    def __getitem__(self, idx):
        NotImplementedError

def json2node(label, scale):
    label = formatnode(label)
    label = formatxy(label, scale)
    return label

def node2json(label, scale):
    label = inverse_formatxy(label, scale)
    label = inverse_formatnode(label)
    return label

class Prosate5Crops:
    def __init__(self, image_path, json_path, csv_path):
        #super().__init__(image_path, json_path, scale)

        # only agument training images
        #image_names, self.labels = self.read_csv(train_csv)
        search_path = os.path.join(image_path, '**', '*.jpg')
        #image_names = set(image_names)
        self.image_names = []
        for fp in glob.iglob(search_path, recursive=True):
            #image_name = os.path.basename(fp)
            #if image_name in image_names:
            self.image_names.append(fp)

        self.transforms = self.aug_transforms()
        self.image_path = image_path
        self.json_path = json_path
        self.scale = 1

        self.image_names, self.gs = self.read_csv(csv_path)
        #for image_name in image_names

    def __len__(self):
        return len(self.image_names)

    def aug_transforms(self):
        def empty_trans(image_data, json, image_trans=True):
            return image_data, json

        vflip_trans = partial(flip_transform, axis=0)
        hflip_trans = partial(flip_transform, axis=1)
        flip_trans = [
            vflip_trans,
            hflip_trans,
        ]
        rotate_trans = [
            rotate180_transform,
            rotate270_transform,
            rotate90_transform,
        ]

        res = [empty_trans]
        res.append(vflip_trans)
        res.append(hflip_trans)
        res.append(rotate90_transform)
        res.append(rotate180_transform)
        res.append(rotate270_transform)

        for f_trans in flip_trans:
            for rot_trans in rotate_trans:
                res.append(Compose([f_trans, rot_trans]))

        #filter out reduanted
        res = res[:-3]
        del res[6]
        assert len(res) == 8

        return res

    def augment(self, image, label):
        images = []
        labels = []

        for transform in self.transforms:
            print(transform)
            tmp_image = image.copy()

            print('>>>>>')
            tmp_image, tmp_label = transform(tmp_image, label, image_trans=True)
            res = node2json(tmp_label, self.scale)
            labels.append(res)
            images.append(tmp_image)
        return images, labels

    def read_csv(self, csv_fp):
        image_names = []
        labels = []
        with open(csv_fp) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image_name, label = row['image name'], row['gleason score']
                image_names.append(image_name)
                labels.append(label)

        return image_names, labels

    def read_json(self, json_path):
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

    def __getitem__(self, idx):
        image_fp = self.image_names[idx]
        image_name = os.path.basename(image_fp)
        #json_name = image_name.replace('.jpg', '.json')
        json_name = image_name.replace('.jpg', '.json')
        image_fp = os.path.join(self.image_path, image_name)
        print(image_fp)
        image = cv2.imread(image_fp, -1)
        #print(image_fp)
        json_fp = os.path.join(self.json_path, json_name)
        json_data = self.read_json(json_fp)

        json_data = json2node(json_data, self.scale)
        images, labels = self.augment(image, json_data)
        #label = self.labels[idx]
        #image_fp = iamge_fp.replace('.', '_aug_{}.'format(idx))
        gs = self.gs[idx]
        if int(gs) <= 6:
            gs = 1
        elif int(gs) > 6:
            gs = 2
        else:
            ValueError('wrong gleason socre value')

        image_fp = image_fp.replace('.', '_grade_{}.'.format(gs))

        return images, labels, image_fp


class BACHAug:
    def __init__(self, image_path, json_path):
        #super().__init__(image_path, json_path, scale)

        # only agument training images
        #image_names, self.labels = self.read_csv(train_csv)
        search_path = os.path.join(image_path, '**', '*.tif')
        #image_names = set(image_names)
        self.image_names = []
        for fp in glob.iglob(search_path, recursive=True):
            #image_name = os.path.basename(fp)
            #if image_name in image_names:
            self.image_names.append(fp)

        self.transforms = self.aug_transforms()
        self.image_path = image_path
        self.json_path = json_path
        self.scale = 1

        #self.image_names, self.gs = self.read_csv(csv_path)
        #for image_name in image_names

    def __len__(self):
        return len(self.image_names)

    def aug_transforms(self):
        def empty_trans(image_data, json, image_trans=True):
            return image_data, json

        vflip_trans = partial(flip_transform, axis=0)
        hflip_trans = partial(flip_transform, axis=1)
        flip_trans = [
            vflip_trans,
            hflip_trans,
        ]
        rotate_trans = [
            rotate180_transform,
            rotate270_transform,
            rotate90_transform,
        ]

        res = [empty_trans]
        res.append(vflip_trans)
        res.append(hflip_trans)
        res.append(rotate90_transform)
        res.append(rotate180_transform)
        res.append(rotate270_transform)

        for f_trans in flip_trans:
            for rot_trans in rotate_trans:
                res.append(Compose([f_trans, rot_trans]))

        #filter out reduanted
        res = res[:-3]
        del res[6]
        assert len(res) == 8

        return res

    def augment(self, image, label):
        images = []
        labels = []

        for transform in self.transforms:
            print(transform)
            tmp_image = image.copy()

            print('>>>>>')
            tmp_image, tmp_label = transform(tmp_image, label, image_trans=True)
            res = node2json(tmp_label, self.scale)
            labels.append(res)
            images.append(tmp_image)
        return images, labels

#    def read_csv(self, csv_fp):
#        image_names = []
#        labels = []
#        with open(csv_fp) as csvfile:
#            reader = csv.DictReader(csvfile)
#            for row in reader:
#                image_name, label = row['image name'], row['gleason score']
#                image_names.append(image_name)
#                labels.append(label)
#
#        return image_names, labels

    def read_json(self, json_path):
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

    def cls_id(self, image_fp):
        image_name = os.path.basename(image_fp)
        if 'b' in image_name:
            return 1
        if 'n' in image_name:
            return 2
        if 'is' in image_name:
            return 3
        if 'iv' in image_name:
            return 4
        raise ValueError('wrong image name')

    def __getitem__(self, idx):
        image_fp = self.image_names[idx]
        image_name = os.path.basename(image_fp)
        #json_name = image_name.replace('.jpg', '.json')
        json_name = image_name.replace('.tif', '.json')
        image_fp = os.path.join(self.image_path, image_name)
        print(image_fp)
        image = cv2.imread(image_fp)
        #print(image_fp)
        json_fp = os.path.join(self.json_path, json_name)
        print(json_fp)
        json_data = self.read_json(json_fp)

        json_data = json2node(json_data, self.scale)
        images, labels = self.augment(image, json_data)
        #label = self.labels[idx]
        #image_fp = iamge_fp.replace('.', '_aug_{}.'format(idx))
        #gs = self.gs[idx]
        #if int(gs) <= 6:
        #    gs = 1
        #elif int(gs) > 6:
        #    gs = 2
        #else:
        #    ValueError('wrong gleason socre value')
        cls_id = self.cls_id(image_fp)

        image_fp = image_fp.replace('.', '_grade_{}.'.format(cls_id))

        return images, labels, image_fp

class Prosate5CropsTest:
    def __init__(self, image_path, json_path, csv_path):
        #super().__init__(image_path, json_path, scale)

        # only agument training images
        #image_names, self.labels = self.read_csv(train_csv)
        search_path = os.path.join(image_path, '**', '*.jpg')
        #image_names = set(image_names)
        self.image_names = []
        for fp in glob.iglob(search_path, recursive=True):
            #image_name = os.path.basename(fp)
            #if image_name in image_names:
            self.image_names.append(fp)

        self.transforms = self.aug_transforms()
        self.image_path = image_path
        self.json_path = json_path
        self.scale = 1

        self.gts = self.read_csv(csv_path)
        #for image_name in image_names

    def __len__(self):
        return len(self.image_names)

    def aug_transforms(self):
        def empty_trans(image_data, json, image_trans=True):
            return image_data, json

        vflip_trans = partial(flip_transform, axis=0)
        hflip_trans = partial(flip_transform, axis=1)
        flip_trans = [
            vflip_trans,
            hflip_trans,
        ]
        rotate_trans = [
            rotate180_transform,
            rotate270_transform,
            rotate90_transform,
        ]

        res = [empty_trans]
        res.append(vflip_trans)
        res.append(hflip_trans)
        res.append(rotate90_transform)
        res.append(rotate180_transform)
        res.append(rotate270_transform)

        for f_trans in flip_trans:
            for rot_trans in rotate_trans:
                res.append(Compose([f_trans, rot_trans]))

        #filter out reduanted
        res = res[:-3]
        del res[6]

        assert len(res) == 8

        return res

    def augment(self, image, label):
        images = []
        labels = []

        for transform in self.transforms:
            print(transform)
            tmp_image = image.copy()

            print('>>>>>')
            tmp_image, tmp_label = transform(tmp_image, label, image_trans=True)
            res = node2json(tmp_label, self.scale)
            labels.append(res)
            images.append(tmp_image)
        return images, labels

    def read_csv(self, csv_fp):
        #image_names = []
        #labels = []
        res = {}
        with open(csv_fp) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image_name, label = row['image name'], row['gleason score']
                #image_names.append(image_name)
                #labels.append(label)
                res[image_name] = label

        return res

    def read_json(self, json_path):
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

    def __getitem__(self, idx):
        image_fp = self.image_names[idx]
        image_name = os.path.basename(image_fp)
        #json_name = image_name.replace('.jpg', '.json')
        json_name = image_name.replace('.jpg', '.json')
        image_fp = os.path.join(self.image_path, image_name)
        print(image_fp)
        image = cv2.imread(image_fp, -1)
        #print(image_fp)
        json_fp = os.path.join(self.json_path, json_name)
        json_data = self.read_json(json_fp)

        json_data = json2node(json_data, self.scale)
        images, labels = self.augment(image, json_data)
        #label = self.labels[idx]
        #image_fp = iamge_fp.replace('.', '_aug_{}.'format(idx))
        gs = self.gts[image_name]
        if int(gs) <= 6:
            gs = 1
        elif int(gs) > 6:
            gs = 2
        else:
            ValueError('wrong gleason socre value')

        image_fp = image_fp.replace('.', '_grade_{}.'.format(gs))

        return images, labels, image_fp

class Prosate:
    def __init__(self, image_path, json_path, train_csv):
        #super().__init__(image_path, json_path, scale)

        # only agument training images
        image_names, self.labels = self.read_csv(train_csv)
        search_path = os.path.join(image_path, '**', '*.jpg')
        image_names = set(image_names)
        self.image_names = []
        for fp in glob.iglob(search_path, recursive=True):
            image_name = os.path.basename(fp)
            if image_name in image_names:
                self.image_names.append(fp)

        self.transforms = self.aug_transforms()
        self.image_path = image_path
        self.json_path = json_path
        self.scale = 1
        #for image_name in image_names

    def __len__(self):
        return len(self.image_names)

    def aug_transforms(self):
        def empty_trans(image_data, json, image_trans=True):
            return image_data, json

        vflip_trans = partial(flip_transform, axis=0)
        hflip_trans = partial(flip_transform, axis=1)
        flip_trans = [
            vflip_trans,
            hflip_trans,
        ]
        rotate_trans = [
            rotate180_transform,
            rotate270_transform,
            rotate90_transform,
        ]

        res = [empty_trans]
        res.append(vflip_trans)
        res.append(hflip_trans)
        res.append(rotate90_transform)
        res.append(rotate180_transform)
        res.append(rotate270_transform)

        for f_trans in flip_trans:
            for rot_trans in rotate_trans:
                res.append(Compose([f_trans, rot_trans]))

        #filter out reduanted
        res = res[:-3]
        del res[6]
        assert len(res) == 8

        return res

    def augment(self, image, label):
        images = []
        labels = []

        for transform in self.transforms:
            print(transform)
            tmp_image = image.copy()

            print('>>>>>')
            tmp_image, tmp_label = transform(tmp_image, label, image_trans=True)
            res = node2json(tmp_label, self.scale)
            labels.append(res)
            images.append(tmp_image)
        return images, labels

    def read_csv(self, csv_fp):
        image_names = []
        labels = []
        with open(csv_fp) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image_name, label = row['image name'], row['gleason score']
                image_names.append(image_name)
                #labels.append(label)
                cls_id = int(label) - 5
                labels.append(cls_id)

        return image_names, labels

    def read_json(self, json_path):
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

    def __getitem__(self, idx):
        image_fp = self.image_names[idx]
        image_name = os.path.basename(image_fp)
        #json_name = image_name.replace('.jpg', '.json')
        json_name = image_name.replace('.jpg', '.json')
        #image_fp = os.path.join(self.image_path, image_name)
        image = cv2.imread(image_fp, -1)
        #print(image_fp)
        json_fp = os.path.join(self.json_path, json_name)
        json_data = self.read_json(json_fp)

        json_data = json2node(json_data, self.scale)
        images, labels = self.augment(image, json_data)
        label = self.labels[idx]
        image_fp = image_fp.replace('.', '_grade_{}.'.format(label))
        #image_fp = iamge_fp.replace('.', '-{}.'format(idx))

        return images, labels, image_fp





class ECRC(BaseWSI):
    def __init__(self, label_path, json_path):
        super().__init__(label_path, json_path)
        self.scale = 2
        self.transforms = self.uniform_transforms()
        #self.image_prefixes = self.image_prefixes * len(self.transforms)

#        print(self.image_prefixes[:30])
#        print(self.transforms[:30])
#        for i, z in zip(self.image_prefixes[:30], self.transforms[:30]):
#            print(i, z)
#
        #assert len(self.transforms) == len(self.image_prefixes)

    def uniform_transforms(self):
        def empty_trans(image_data, json, image_trans=True):
            return image_data, json

        vflip_trans = partial(flip_transform, axis=0)
        hflip_trans = partial(flip_transform, axis=1)
        flip_trans = [
            vflip_trans,
            hflip_trans,
        ]
        rotate_trans = [
            rotate180_transform,
            rotate270_transform,
            rotate90_transform,
        ]

        res = [empty_trans]
        res.append(vflip_trans)
        res.append(hflip_trans)
        res.append(rotate90_transform)
        res.append(rotate180_transform)
        res.append(rotate270_transform)

        for f_trans in flip_trans:
            for rot_trans in rotate_trans:
                res.append(Compose([f_trans, rot_trans]))

        #filter out reduanted
        res = res[:-3]
        del res[6]
        assert len(res) == 8

        return res

    def __getitem__(self, idx):
        p = self.image_prefixes[idx]

        image = self.image_dataset.whole_file(p)
        base_name = os.path.basename(p)
        label = self.json_dataset.whole_file(base_name)
        #count += len(label)
        #print(label[10])
        #import time
        #start = time.time()
        #label = formatnode(label)
        #label = formatxy(label, 2)
        #print(label[0])
        label = json2node(label, self.scale)
        #print(label[0])

        images, labels = [], []

        if self.transforms is not None:
            for transform in self.transforms:
                print(transform)
                tmp_image = image.copy()

            #if random.random() > 0.5:
                print('>>>>>')
                #count = 0
                tmp_image, tmp_label = transform(tmp_image, label, image_trans=True)
                #for node in label:
                        #print(node)

                    #cen = node.centroid
                    #ori_image = image.copy()
                    #image, cen = transform(image, cen)
                    #bbox = node.bbox
                    #_, b1 = transform(ori_image, bbox[:2], image_trans=False)
                    #_, b2 = transform(ori_image, bbox[2:], image_trans=False)
                    #bbox = [*b1, *b2]
                    #contour = None
                #res.append(Node(centroid=cen, bbox=bbox, contour=contour))
                res = node2json(tmp_label, self.scale)
                labels.append(res)
                #print(tmp_image.shape, "???????", type(tmp_image))
                images.append(tmp_image)

            #label = node2json(label, self.scale)

        path = self.image_dataset.prefix2image_name(p)
        assert len(images) == len(labels)
        return images, labels, path

class Compose:
    def __init__(self, trans):
        self.trans = trans

    #def __str__(self):
    #    res = ''
    #    for trans in self.trans:
    #        res += ' ' + trans.__name__

    #    return  res

    def __call__(self, image, label, image_trans=True):
        #if not image_trans:
        image_cp = image.copy()

        for trans in self.trans:
            print(trans)
            image_cp, label = trans(image_cp, label, image_trans=image_trans)

        if image_trans:
            return image_cp, label
        else:
            return image, label



#def all_transforms():
#    vflip_trans = partial(flip_transform, axis=0)
#    hflip_trans = partial(flip_transform, axis=1)
#    return [
#        vflip_trans,
#        hflip_trans,
#        rotate180_transform,
#        rotate270_transform,
#        rotate90_transform,
#    ]

def gen_transforms():
    rotate_trans = RandomChoice([
        rotate90_transform,
        rotate180_transform,
        rotate270_transform
    ])

    vflip_trans = partial(flip_transform, axis=0)
    hflip_trans = partial(flip_transform, axis=1)

    return [rotate_trans, vflip_trans, hflip_trans]










        #image = draw_nuclei(image, label)
        ##cv2.imwrite('hhhh1.jpg', image)
        ##image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
        ##cv2.imwrite('hhhh.jpg', image)
        #label = inverse_formatxy(label, 2)
        #label = inverse_formatnode(label)
        ##print(time.time() - start)
        ##print(label[10])
        #import sys; sys.exit()
    #for l in label:
        #print()
    #print(len(label))
    #print(label)
    #print(image.shape)

def get_image_fp_ecrc(path, image_save_path, idx):
    img_save_fp = os.path.join(image_save_path, *path.split(os.sep)[1:])
    #img_save_fp = img_save_fp.split('.')[0] + '-{}.json'.format(idx)
    img_save_fp = img_save_fp.replace('.', '-{}.'.format(idx))

    return img_save_fp

def get_json_fp_ecrc(path, json_save_path, idx):
    basename = os.path.basename(path)
    basename = basename.replace('.png', '.json')
    path = os.path.join(json_save_path, basename)
    path = path.replace('.', '-{}.'.format(idx))
    return path

def get_image_fp_prostate(path, image_save_path, idx):
    base_name = os.path.basename(path)
    img_save_fp = os.path.join(image_save_path, base_name)
    img_save_fp = img_save_fp.replace('.', '-{}.'.format(idx))
    return img_save_fp

def get_image_fp_bachaug(path, image_save_path, idx):
    basename = os.path.basename(path)
    image_save_fp = os.path.join(image_save_path, basename)
    img_save_fp = image_save_fp.replace('.', '_aug_{}.'.format(idx))
    return img_save_fp

def get_json_fp_bachaug(path, json_save_path, idx):
    basename = os.path.basename(path)
    basename = basename.replace('.tif', '.json')
    json_save_fp = os.path.join(json_save_path, basename)
    json_save_fp = json_save_fp.replace('.', '_aug_{}.'.format(idx))

    return json_save_fp



def get_image_fp_prostate5crops(path, image_save_path, idx):
    base_name = os.path.basename(path)
    img_save_fp = os.path.join(image_save_path, base_name)
    img_save_fp = img_save_fp.replace('.', '_aug_{}.'.format(idx))
    return img_save_fp

def get_json_fp_prostate5crops(path, json_save_path, idx):
    basename = os.path.basename(path)
    basename = basename.replace('.jpg', '.json')
    json_save_fp = os.path.join(json_save_path, basename)
    json_save_fp = json_save_fp.replace('.', '_aug_{}.'.format(idx))

    return json_save_fp

def get_json_fp_prostate(path, json_save_path, idx):
    basename = os.path.basename(path)
    basename = basename.replace('.jpg', '.json')
    json_save_fp = os.path.join(json_save_path, basename)
    json_save_fp = json_save_fp.replace('.', '-{}.'.format(idx))

    return json_save_fp

#image_path = 'test_can_be_del3'
#json_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Json/EXtended_CRC_Mask/'
#image_save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Images_Aug'
#json_save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Json_Aug'
#dataset = ECRC(image_path, json_path)

if __name__ == '__main__':

    #image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/images/'
    #json_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/json_labels/json/'
    #image_save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images_Aug/train/'
    #json_save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json_Aug/train'
    #dataset = Prosate(image_path, json_path, '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/Gleason_masks_train.csv')

    #image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/5Crops'
    #json_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/5Crops'
    #image_save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/5Crops_Aug'
    #json_save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/5Crops_Aug'
    #csv_file = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/5Crops.csv'
    #dataset = Prosate5Crops(image_path, json_path, csv_file)

    #image_save_path = 'tmp2'
    #json_save_path = 'tmp2'
    #image_path = '/home/baiyu/HGIN/immage_test'
    #json_path = '/home/baiyu/HGIN/json_test'
    #image_save_path = '/data/smb/syh/tmp'
    #json_save_path = '/data/smb/syh/tmp/'
    #csv_file = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/5Crops.csv'
    #dataset = Prosate5CropsTest(image_path, json_path, csv_file)

    image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Images'
    json_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Json/json'
    image_save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Images_Aug'
    json_save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Json_Aug'
    #csv_file = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/5Crops.csv'
    #dataset = Prosate5Crops(image_path, json_path)
    dataset = BACHAug(image_path, json_path)

    #image_count = 0
    print(len(dataset))
    for images, labels, path in dataset:
        for idx, (image, label) in enumerate(zip(images, labels)):
            #cv2.imwrite('tmp/{}_{}.jpg'.format(image_count, trans_count), image)
            #json_name = image_name.replace('.', '-{}.'.format(idx))
            #json_fp = image_name.split('.')[0] + '-{}.json'.format(idx)
            #json_fp = os.path.join(json_save_path, json_fp)
            #img_fp = img_save_fp.replace('.', '-{}.'.format(idx))
            #img_fp = get_image_fp_prostate(path, image_save_path, idx)
            #img_fp = get_image_fp_ecrc(path, image_save_path, idx)
            #json_fp = get_json_fp_ecrc(path, json_save_path, idx)
            #img_fp = get_image_fp_prostate5crops(path, image_save_path, idx)
            #json_fp = get_json_fp_prostate5crops(path, json_save_path, idx)
            img_fp = get_image_fp_bachaug(path, image_save_path, idx)
            json_fp = get_json_fp_bachaug(path, json_save_path, idx)


            os.makedirs(os.path.dirname(img_fp), exist_ok=True)
            print('saving image to {}...'.format(img_fp))
            cv2.imwrite(img_fp, image)
            os.makedirs(os.path.dirname(json_fp), exist_ok=True)
            print('saving json file to {}...'.format(json_fp))
            save_json(json_fp, label)
        #import sys;sys.exit()

            #print(img_fp)
            #print(json_fp)
            #print(os.path.join(json_save_path, json_name))
            #print(os.path.join(img_save_fp.replace('.', '-{}.'.format(idx))))
            #save_json(os.path.join(json_save_path, json_name.format(trans_count)).format(image_count, trans_count), label)
            #node = json2node(label, scale=1)
            #print(image.shape, 'cccccccccccccc', image.dtype)
            #image = draw_nuclei(image.astype('uint8'), node)
            #cv2.imwrite('tmp1/{}_{}.jpg'.format(image_count, trans_count), image)
            #image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            #cv2.imwrite('tmp2/{}_s.jpg'.format(idx), image)






    #ppp = 'tmp/1_5.jpg'
    #ppp1 = 'tmp/1_5.json'
    #image = cv2.imread(ppp)
    #json_data = json.load(open(ppp1, 'r'))
    #res = []
    #for k, v in json_data['nuc'].items():
    #    res.append(v)
    #
    #nodes = json2node(res, 2)
    #image = draw_nuclei(image, nodes)
    ##
    ##cv2.imwrite('hhhh1_.jpg', image)
    #image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
    #cv2.imwrite('hhhh_5.jpg', image)



    #test_rorate90()
    #print(count / 300)
