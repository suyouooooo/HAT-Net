import torch_geometric
import torch
import numpy as np
from torch_cluster import grid_cluster
import math
import re

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

#class Dummy:
#    def __init__(self, aa):
#        self.num = aa

#from concurrent.futures import ThreadPoolExecutor
#import time
#start = time.time()
#for i in range(100000):
#    a = ThreadPoolExecutor(8)
#
#print(time.time() - start)






#class Data:
#    def __init__(self):
#        self.res = []
#        for i in range(1000):
#            self.res.append(Dummy(i))
#
#    def __len__(self):
#        return len(self.res)
#
#    def __getitem__(self, idx):
#        return self.res[idx]
#
#
#def ccc(batch):
#    return batch
#dataset = Data()
##for i in dataset:
##    print(i.num)
#dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=16, collate_fn=lambda x : x)
#
#for i in dataloader:
#    print(len(i))
#    print(i[3].num)
#

import torch_geometric

import glob
import os
import csv
import torch


def read_csv(csv_fp):
    image_names = []
    labels = []
    res = {}
    with open(csv_fp) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_name, label = row['image name'], row['gleason score']
            res[image_name.split('.')[0]] = label

    return res

csv_file = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/5Crops.csv'
def fix_label(labels, fp):
        base_name = os.path.basename(fp)
        prefix, ext = base_name.split('_grade_')
        label = int(labels[prefix])
        if label >= 6:
            label = 2
        else:
            label = 1

        ext = str(label) + ext[1:]
        prefix = fp.split('_grade_')[0]
        fp = prefix + '_grade_' + ext
        return fp

path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Aug_CGC16dim/'
res = read_csv(csv_file)

def fix_name(basename, gt):
    new_grade = '_grade_{}'.format(gt + 1)
    old_grade = re.search(r'_grade_[0-9]', basename).group()
    #print(old_grade, new_grade)
    return basename.replace(old_grade, new_grade)


    #if grade in basename:
    #    return basename



#print(res)
#save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Aug_CGC16dim_Fix_label/'
#count = 0
#for i in glob.iglob(os.path.join(path, '**', '*.pt'), recursive=True):
#    basename = os.path.basename(i)
#    gleason_score = int(res[basename.split('_grade_')[0]])
#    data = torch.load(i)
#    if gleason_score >= 6:
#        cls_id = 1 # high-risk
#    else:
#        cls_id = 0 # low-risk
#        #count += 1
#
#    new_basename = fix_name(basename, cls_id)
#    data.y = torch.tensor(cls_id)
#    #print(basename, new_basename)
#    data.path = new_basename
#    #print(data.path, data.y)
#    torch.save(data, os.path.join(save_path, new_basename))

    #print(data.path)

#print(count)
    #print(data.y)
    #if data.y != cls_id:
    #    #print(data.y.item(), cls_id)
    #    data.y = torch.tensor(cls_id)
    #    print(data.y)



import re
import os
import glob


#def cls_id(image_fp):
        #image_name = os.path.basename(image_fp)
        #'Normal, Benign, in situ and Invasive {0,1,2,3}'
        #if 'b' in image_name:
        #    return 1
        #if 'n' in image_name:
        #    return 2
        #if 'is' in image_name:
        #    return 3
        #if 'iv' in image_name:
        #    return 4
        #raise ValueError('wrong image name')
def rename(src_path):
    #grade = re.search(r'_grade_[0-9]', src_path).group()
    #grade_id = int(grade[-1])
    #dirname = os.path.dirname(src_path)
    basename = os.path.basename(src_path)
    if '_grade_1' in basename:
        dest_path = basename.replace('_grade_1', '_grade_2')
    else:
        dest_path = basename.replace('_grade_2', '_grade_1')

    return dest_path

#data_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Json_Aug_withtypes'
#for i in glob.iglob(os.path.join(data_folder, '**', '*.json'), recursive=True):
#    #before = os.path.basename(i)
#    #basename = rename(i)
#    #if before != basename:
#        #print(before, basename)
#
#    basename = os.path.join
#labels = read_csv(csv_file)
#count = 0
#for i in glob.iglob(os.path.join('/data/hdd1/by/HGIN/acc872_prostate_5cropsAug_fix_label', '**', '*.pt'), recursive=True):
#    before = i
#    after = fix_label(labels, i)
#
#    label = int(after.split('_grade_')[1][0])
#
#    data = torch.load(before)
#
#    if before != after:
#        #print('{} --> {}'.format(before, after))
#        data = torch.load(before)
#        #data.path
#        os.remove(before)
#        data.y = label - 1
#        data.path = fix_label(labels, data.path)
#        #print(data.path, label, data.y)
#        #data.path =
#        #print(data)
#        torch.save(data, after)
#        count += 1
#
#print(count)


#small_value = 0
#samll_count = 0
#total = 0
##for i in glob.iglob(os.path.join('/data/hdd1/by/HGIN/acc872_prostate_5cropsAug_fix_label', '**', '*.pt'), recursive=True):
#for i in glob.iglob(os.path.join('/data/smb/syh/PycharmProjects/CGC-Net/data_res50/proto/fix_fuse_cia_knn/0', '**', '*.pt'), recursive=True):
##for i in glob.iglob(os.path.join('/data/hdd1/by/HGIN/ttt_del1', '**', '*.pt'), recursive=True):
#    data = torch.load(i)
#    #print(data.x.shape)
#    mask = data.x <= 1e-7
#    small_value += mask.sum()
#    total += data.x.numel()
#    #print(small_value)
#    #print(mask)
#    #print(data.x[~mask])
#    #import sys; sys.exit()
#
#print(small_value / total)
#
def read_csv(csv_fp):
        image_names = []
        labels = []
        res = {}
        with open(csv_fp) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                case, label = row['case'], row['class']
                image_name = 'test{}.tif'.format(case)
                label = int(label) + 1
                print('before', label)
                if label == 1:
                    label = 2
                elif label == 2:
                    label = 1
                print('after', label)
                res[image_name] = label

        return res

#csv_file = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/pred.csv'
#print(read_csv(csv_file))

def num_nodes(path):
    search_path = os.path.join(path, '**', '*.pt')
    cell_count = 0
    count = 0
    for i in glob.iglob(search_path, recursive=True):
        #print(i)
        data = torch.load(i)
        #print(data.x.shape)
        cell_count += data.x.shape[0]
        count += 1

    print(cell_count / count)



def degree(path):
    from collections import Counter
    search_path = os.path.join(path, '**', '*.pt')
    counter = Counter()
    for i in glob.iglob(search_path, recursive=True):
        #print(i)
        data = torch.load(i)
        print(data.num_edges / data.num_nodes)
        row, col = data.edge_index
        #print(row.shape, 1)
        #print(col.shape, 12)
        edge_idx = torch_geometric.utils.degree(col, data.x.size(0))
        #edge_idx = torch_geometric.utils.degree(row)
        #edge_idx = torch_geometric.utils.degree(row)
        #print(edge_idx.shape, 33)
        #print(data.x.shape)
        counter.update(edge_idx.tolist())

    print(counter)

path = '/data/hdd1/by/HGIN/trainbach512/hatnet512dim'
#num_nodes(path)
#print(degree(path))


def slice_image(image, h=4, w=4):
    img_h, img_w = image.shape[:2]

    #print(img_h, h, img_h / 4)
    print(image.shape)
    #assert (img_h / h).is_integer()
    #assert (img_w / w).is_integer()
    patch_h = math.ceil(img_h / h)
    patch_w = math.ceil(img_w / w)
    print(patch_h, patch_w)

    res = []
    for h_idx in range(0, img_h, patch_h):
        for w_idx in range(0, img_w, patch_w):
            #print(h_idx, w_idx)
            patch = image[h_idx: h_idx + patch_h, w_idx: w_idx + patch_w]
            res.append(patch)

    print(len(res))
    return res

#img = np.zeros((7548, 4548, 3))
#print(img.shape)
#res = slice_image(img)
#print(len(res))

class ImageWriter:
    def __init__(self, save_path):
        self.save_path = save_path
        self.pool = ThreadPoolExecutor.pool(8)


        #for idx, patch in enumerate(patches):

    def write(self, path, patches):
        idx = range(len(patches))
        path = repeat(path)
        self.pool.map(self.write_file, zip(dix, path, patches))

    def write_file(self, idx, i, patch):
            basename = os.path.basename(i)
            basename = basename.replace('.png', '_{}.png'.format(idx))
            #print(basename)
            path = os.path.join(self.save_path, basename)
            #print(path, patch.shape)
            cv2.imwrite(path, patch)


import cv2
import math
#src_path = '/data/smb/syh/colon_dataset/CRC_Dataset'
src_path = '/data/smb/数据集/结直肠/病理学/Extended_CRC/Original_Images/'
##save_path = 'crc_sliced'
#save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/SliceImage'
save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/SliceImage'
count = 0
#imgwriter = ImageWriter(save_path)
#for i in glob.iglob(os.path.join(src_path, '**', '*.png'), recursive=True):
#    count += 1
#    image = cv2.imread(i)
#    patches = slice_image(image)
#    #imgwriter.write(i, patches)
#    for idx, patch in enumerate(patches):
#        basename = os.path.basename(i)
#        basename = basename.replace('.png', '_{}.png'.format(idx))
#        #print(basename)
#        path = os.path.join(save_path, basename)
#        #print(path, patch.shape)
#        cv2.imwrite(path, patch)
#        #print(path)
#
#import sys; sys.exit()


#import torch
#a = torch.tensor([[0, 3, 1],
#                  [1, 1, 0 ]])
#from torch_geometric.utils import to_dense_adj
#
#print(to_dense_adj(a))
import cv2

def add_image_shape(image_path, pt_path):
    for i in glob.iglob(os.path.join(pt_path, '**', '*.pt'), recursive=True):
        #print(i)
        #basename = os.path.basename(i)
        #image_name = basename.split('.')[0] + '.tif'
        #image_name = basename.split('_grade_')[0] + '.tif'
        #image_fp = os.path.join(image_path, image_name)
        #print(image_fp, i)
        #print(image_fp, i)
        #image_size = cv2.imread(image_fp).shape
        #print(image_fp)
        data = torch.load(i)
        #data.image_size = image_size
        data.degree = torch_geometric.utils.degree(data.edge_index[1])
        #print(data.degree.max())
        image_size = data.image_size
        h, w = image_size[:2]
        h = int(math.ceil(h / 2))
        w = int(math.ceil(w / 2))
        data.cluster = grid_cluster(data.pos, torch.tensor([h, w]).to(data.x.device))
        #print(data.cluster.min(), data.cluster.max())
        #print(data.cluster)
        #print(data)

        #print(data.image_size)
        torch.save(data, i)

def read_csv(csv_fp):
    image_names = []
    labels = []
    res = {}
    with open(csv_fp) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_name, label = row['image name'], row['gleason score']
            res[image_name.split('.')[0]] = label

    return res
image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Images_Aug'
#image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Images/test'
pt_path = 'trainbach512/hatnet512dim/'
#pt_path = 'testbach512/hatnet512dim/'
#add_image_shape(image_path, pt_path)
path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/Gleason_masks_test_pathologist1.csv'


#a = num_nodes('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Aug_CGC16dim/')
#a = num_nodes('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops')
#print(a)
#res = read_csv(path)
#print(res)

import os

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

def remove_files(path):

    import shutil
    shutil.rmtree(path)
    #for i in glob.iglob(os.path.join(path, '**', '*.jpg')):
        #print(i)
        #break


print('here')
#remove_files('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/Patches_Binary')
#a = num_nodes('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Aug_CGC16dim/')
#a = num_nodes('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Cell_Graph/1792_Avg_64/')
#print(a)
#print(remove_files('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/Patches_Binary') / 1024 / 1024 / 1024, 'Gbytes')

class PanNuke:
    def __init__(self, path):
        self.path = path
        search_path = os.path.join(path, '*', '*.npy')
        for i in glob.iglob(search_path, recursive=True):
            if 'images.npy' in os.path.basename(i):
                self.images = np.load(i, mmap_mode='r')
            if 'masks.npy' in os.path.basename(i):
                self.masks = np.load(i, mmap_mode='r')
            if 'types.npy' in os.path.basename(i):
                self.types = np.load(i, mmap_mode='r')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx], self.types[idx]

def dilate(mask):
    kernel = np.ones((3, 3), np.uint8)
    #print(mask.shape, 9999999)
    #print(mask.dtype)
    #dal = cv2.dilate(mask.astype('uint8'), kernel, iterations=4)
    dal = cv2.dilate(mask, kernel, iterations=4)
    #print(np.unique(dal))
    #print(dal.shape, mask.shape)
    tmp_mask = np.bitwise_xor(mask > 0, dal > 0)
    #print(np.unique(mask), 111)
    #tmp = image.copy()
    #print(tmp_mask.shape, 11111111)
    #tmp[tmp_mask != 0, :] = 0
    return tmp_mask

def overlay(image, mask):
    image = image.copy()
    #img.setflags(write=1)
    color0 = (255, 0, 0)[::-1]
    color1 = (255, 94, 0)[::-1]
    color2 = (103, 181, 85)[::-1]
    color3 = (255, 255, 49)[::-1]
    color4 = (255, 0, 252)[::-1]
    #color5 = (145, 132, 80)[::-1]
    mask0 = dilate(mask[:, :, 0])
    mask1 = dilate(mask[:, :, 1])
    mask2 = dilate(mask[:, :, 2])
    mask3 = dilate(mask[:, :, 3])
    mask4 = dilate(mask[:, :, 4])
    #mask5 = dilate(mask[:, :, 5]) #bg
    #print(mask.shape)

    image[mask0] = color0
    image[mask1] = color1
    image[mask2] = color2
    image[mask3] = color3
    image[mask4] = color4
    #image[mask5] = color5

    return image


path = '/data/smb/syh/colon_dataset/PanukeEx/cancer-instance-segmentation-and-classification-2'
dataset = PanNuke(path)

print(len(dataset))

count = 0
for i in dataset:
    count += 1
    #i = dataset[144]
    image, mask, type_id = i
    #print(image.shape)
    #print(mask.shape)
    image = overlay(image, mask)
    #print(type_id)
    cv2.imwrite('pannuke_overlay/{}_{}.jpg'.format(type_id, count), image)
    #break
