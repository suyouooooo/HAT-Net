import math
from collections import namedtuple
#import torch_geometric
#from torch_geometric.data import Data


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
    new_x = h - y - 1

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
    if axis == 0:
        new_y = 2 * or_y - y
        assert new_y >= 0
        return (x, new_y)
    elif axis == 1:
        new_x = 2 * or_x - x
        assert new_x >= 0
        return (new_x, y)
    else:
        raise ValueError('axis should only be 1 or 0')

def rotate90_transform(image_data, nodes):
    """
        rotate clockwise 90 degrees
        point: (x, y)
    """
    image_data = cv2.rotate(image_data, cv2.cv2.ROTATE_90_CLOCKWISE)
    res = []
    for node in nodes:
        cen = rotate90(node.centeroid)
        b1 = rotate90(node.bbox[:2])
        b2 = rotate90(node.bbox[2:])
        res.append(Node(centroid=cen, bbox=[*b1, *b2]))

    return image_data, res

def rotate180_transform(image_data, nodes):
    image_data, nodes = rotate90_transform(image_data, nodes)
    return image_data, nodes

def rotate270_transform(image_data, nodes):
    image_data, nodes = rotate90_transform(image_data, nodes)
    image_data, nodes = rotate180_transform(image_data, nodes)
    return image_data, nodes

def flip_transform(image_data, nodes, axis):
    """
        nodes: list of nodes
        node bbox: [xmin, ymin, xmax, ymax]
             cen: [x, y]
    """

    y, x = image_data.shape[:2]
    origin = int(x / 2),  int(y / 2)

    #for k, nodes in json_data['nuc'].items():
    res = []
    for node in nodes:
            cen = node.centroid  # (x, y)
            cen = flip(origin, cen, axis=axis)
            b1 = flip(origin, node.bbox[:2], axis=axis)
            b2 = flip(origin, node.bbox[2:], axis=axis)
            res.append(Node(centroid=cen, bbox=[*b1, *b2]))

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



import stich

from stich import JsonDataset, JsonFolder, ImageDataset, ImageFolder


'/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Images/'
json_lists = JsonFolder('')