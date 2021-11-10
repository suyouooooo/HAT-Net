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
    dal = cv2.dilate(mask, kernel, iterations=2)
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
    color0 = (255, 0, 0)[::-1]  # red
    color1 = (255, 94, 0)[::-1]  # orange
    color2 = (103, 181, 85)[::-1] # green
    color3 = (255, 255, 49)[::-1] # yellow
    color4 = (255, 0, 252)[::-1] # purple
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

def pad_patch(min_x, min_y, max_x, max_y, output_size):
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

#def mask_out(image, mask):
def crop_out(image, mask):
    for i in range(mask.shape[2]):
        if i == 5:
            continue

        n_ids = np.unique(mask[:, :, i])
        tmp_img = image.copy()
        #tmp_mask = np.zeros((image.shape[:2]))
        bg_mask = mask[:, :, 5] > 0
        nuclei_mask = mask[:, :, i] > 0
        #tmp_mask[bg_mask & nuclei_mask] = 1
        #tmp_img[bg_mask & nuclei_mask] = 0
        #tmp_img[bg_mask & nuclei_mask] = 0
        tmp_img[~(bg_mask + nuclei_mask)] = 0
        #cv2.imwrite('tmp2/tttt{}.jpg'.format(i), tmp_img)
        if len(n_ids) > 0:
        #print(n_ids)
            for n_id in n_ids:
                if n_id == 0:
                    continue
                #print(i)
                row, col = (mask[:, :, i] == n_id).nonzero()
                min_y = row.min()
                max_y = row.max()
                min_x = col.min()
                max_x = col.max()
                #print(r_min, r_max, c_min, c_max)
                #r_min, c_max,
                min_x, min_y, max_x, max_y = pad_patch(min_x, min_y, max_x, max_y, 64)
                bbox = [min_x, min_y, max_x, max_y]
                patch = tmp_img[min_y : max_y, min_x : max_x]
                print(patch.shape)
                cv2.imwrite(
                    'tmp2/{}_{}.jpg'.format(i, n_id),
                    patch
                )

                #if i == 0 and n_id == 25:
                #    image = cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color=(0, 0, 0), thickness=3)

                #    #cv2.rectangle(image, )
                #    cv2.imwrite('tmp2/{}.jpg'.format(500), image)


                #if i == 2 and n_id == 69:
                #    image = cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color=(0, 0, 0), thickness=3)

                #    #cv2.rectangle(image, )
                #    cv2.imwrite('tmp2/{}.jpg'.format(500), image)



#def crop_consep(image, mask):
    #for i in mask['inst_type']:


path = '/data/smb/syh/colon_dataset/PanukeEx/cancer-instance-segmentation-and-classification-2'
#dataset = PanNuke(path)

#print(len(dataset))

count = 0
#path = '/data/smb/syh/colon_dataset/Kumar/Kumar/train/'
path = '/data/smb/syh/colon_dataset/Kumar/Kumar/train/'
images = os.path.join(path, 'Images')
labels = os.path.join(path, 'Labels')

label = 'TCGA-G9-6348-01Z-00-DX1.mat'
image = 'TCGA-G9-6348-01Z-00-DX1.tif'

image_path = os.path.join(images, image)
label_path = os.path.join(labels, label)
#overlay = '/data/smb/syh/colon_dataset/Kumar/Kumar/train/Overlay/TCGA-G9-6348-01Z-00-DX1.png'
overlay = '/data/smb/syh/colon_dataset/CoNSeP/Train/Overlay/train_23.png'
label_path = '/data/smb/syh/colon_dataset/CoNSeP/Train/Labels/train_23.mat'
print(image_path)
print(label_path)
print(overlay)

#import scipy.io as sio
#mat = sio.loadmat(label_path)
#n_ids = np.unique(mat['inst_map'])
#image = cv2.imread(overlay)
#mask = mat['inst_map']
## bg_mask = mask[:, :] == 0
## nuclei_mask = mask[:, :] > 0
##         #tmp_mask[bg_mask & nuclei_mask] = 1
##         #tmp_img[bg_mask & nuclei_mask] = 0
##         #tmp_img[bg_mask & nuclei_mask] = 0
## tmp_img[~(bg_mask + nuclei_mask)] = 0
#for n in n_ids:
#    if n == 0:
#        continue
#
#    row, col = (mask == n).nonzero()
#    min_y = row.min()
#    max_y = row.max()
#    min_x = col.min()
#    max_x = col.max()
#                #print(r_min, r_max, c_min, c_max)
#                #r_min, c_max,
#    min_x, min_y, max_x, max_y = pad_patch(min_x, min_y, max_x, max_y, 64)
#    bbox = [min_x, min_y, max_x, max_y]
#    patch = image[min_y : max_y, min_x : max_x]
#    #print(patch.shape)
#    cv2.imwrite(
#        'tmp3/{}.jpg'.format(n),
#        patch
#    )



#for i in dataset:
#    #print(i)
#    i = dataset[1000]
#    image, mask, type_id = i
#    image = overlay(image, mask)
#    cv2.imwrite('tmp2/{}.jpg'.format(1000), image)
#    #crop_out(image, mask)
#
#    #mask_out(image, mask)
#    #print(image.shape, mask.shape, type_id)
#    #print(mask.shape)
#    #print(mask[:, :, 0].max())
#    #print(np.unique(mask))
#    break
#for i in dataset:
#    count += 1
#    #i = dataset[144]
#    #print(image.shape)
#    #print(mask.shape)
#    #print(type_id)
#    cv2.imwrite('pannuke_overlay/{}_{}.jpg'.format(type_id, count), image)
    #break

def draw_mask(image, mask):
    #tmp = image.copy()
    tmp = cv2.resize(image, (0, 0), fx=2, fy=2)
    tmp_mask = dilate(mask)
    tmp[tmp_mask] = (0, 255, 0)
    tmp = cv2.resize(tmp, (0, 0), fx=0.5, fy=0.5)

    return tmp

def crop_patches(image, mask):
    image = cv2.resize(image, (0, 0), fx=2, fy=2)
    for n_id in np.unique(mask):
        if n_id == 0:
            continue

        row, col = (mask == n_id).nonzero()
        min_y = row.min()
        max_y = row.max()
        min_x = col.min()
        max_x = col.max()
        #print(r_min, r_max, c_min, c_max)
        #r_min, c_max,
        min_x, min_y, max_x, max_y = pad_patch(min_x, min_y, max_x, max_y, 64)
        bbox = [min_x, min_y, max_x, max_y]
        patch = image[min_y : max_y, min_x : max_x]
        #print(patch.shape)
        cv2.imwrite(
            'tmp2/{}.jpg'.format(n_id),
            patch
        )

def draw_coords(image, coords):
    #cv2.circle(image, tuple(coord[::-1]), 3, (0, 200, 0), cv2.FILLED, 1)
    #image = cv2.resize(image, (0, 0), fx=2, fy=2)
    for coord in coords:
        image = cv2.circle(image, tuple(coord.long().tolist()[::-1]), 8, (50, 255, 255), cv2.FILLED, 1)

    #image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    return image

def draw_edges(image, edges, coords):
    #image = cv2.resize(image, (0, 0), fx=2, fy=2)
    for i in  range(edges.shape[1]):
        start = coords[edges[0, i]]
        end = coords[edges[1, i]]
        #start = start / 2
        #end = end / 2
        #print(start, end)
        image = cv2.line(image, start.long().tolist()[::-1], end.long().tolist()[::-1], color=(0, 255, 0), thickness=3)

    #image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    return image

def crop_patch(image, start, patch_size):
    h, w = start
    patch = image[h : h + patch_size, w : w + patch_size].copy()
    image = cv2.rectangle(image, [w, h], [w + patch_size, h + patch_size], color=(0, 0, 255), thickness=6)
    return image, patch

def vis_graph(image, cg):
    image = draw_edges(image, cg.edge_index, cg.pos)
    image = draw_coords(image, cg.pos)
    return image


#def draw_cg_figure(image, cg, crop_coord, patch_size):
#
#    image = vis_graph(image, cg)
#    h, w = crop_coord
#    first_img, patch = crop_patch(image, (h, w), patch_size)


image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC/fold_1/1_normal/H10-24087_A2H_E_1_1_grade_1_2689_2913.png'
mask_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/proto/mask/CRC/shaban-cia/fold_1/1_normal/H10-24087_A2H_E_1_1_grade_1_2689_2913.npy'
pos_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/proto/coordinate/CRC/fold_1/1_normal/H10-24087_A2H_E_1_1_grade_1_2689_2913.npy'
cg_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_res50/proto/fix_avg_cia_knn/0/fold_1/H10-24087_A2H_E_1_1_grade_1_2689_2913.pt'

import numpy as np
import cv2
mask = np.load(mask_path)
image = cv2.imread(image_path)
orig = image.copy()

#print(image.shape)
#print(mask.shape)

overlay = draw_mask(image, mask)
print(overlay.shape)
first_img, patch = crop_patch(overlay, (1000, 300), 251)
cv2.imwrite('aa.jpg', first_img)
cv2.imwrite('ff.jpg', patch)
#crop_patches(overlay, mask)

mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)
#mask = mask / mask.max() * 255
mask[mask > 0] = 255

#cv2.imwrite('aa1.jpg', mask)

h, w = mask.shape[:2]
#print(h, w)
#patch =
#print(mask.shape)
patch = overlay[h // 2 : , w // 2 : ]
image[:, :, 0] = mask
image[:, :, 1] = mask
image[:, :, 2] = mask
#print(np.unique(patch))
image[h // 2 :, w // 2 :] = patch
cv2.imwrite('aa1.jpg', image)


#pos = np.load(pos_path)
cg = torch.load(cg_path)

image = cv2.imread(image_path)
#image = cv2.resize(image, (0, 0), fx=2, fy=2)
#draw_edges = draw_edges(image, cg.edge_index, cg.pos)

#draw_coords = draw_coords(image, cg.pos)

#cv2.imwrite('hello1.jpg', first_img)
#cv2.imwrite('hello_patch1.jpg', patch)


# crc dataset
#image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC/fold_2/3_high_grade/H09-24359_A2H_E_1_6_grade_3_2689_3137_180.png'
#graph_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_res50/proto/fix_avg_cia_knn/0/fold_2/H09-24359_A2H_E_1_6_grade_3_2689_3137_180.pt'
#image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Images/fold_3/3_high_grade/Grade3_Patient_172_9_grade_3_row_2688_col_4256.png'
#image_path = 'ecrc.jpg'
#graph_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Cell_Graph/1792_avg_64/proto/fix_avg_cia_knn/0/fold_3/Grade3_Patient_172_9_grade_3_row_2688_col_4256.pt'
#image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Images/train/is082.tif'
#graph_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/Aug/hatnet512dim/is082_grade_3_aug_0.pt'
image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/5Crops/ZT80_38_C_8_8_crop_4.jpg'
graph_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops/ZT80_38_C_8_8_crop_4_grade_2.pt'

image = cv2.imread(image_path)
cg = torch.load(graph_path)
image = vis_graph(image, cg)
print(image.shape)

#first_img, patch = crop_patch(image, (1000, 300), 256)
first_img, patch = crop_patch(image, (530, 500), 256)


#cv2.imwrite('crc_vis.jpg', first_img)
#cv2.imwrite('crc_vis.jpg', first_img)
print(first_img.shape, 1111)
cv2.imwrite('uzh_vis.jpg', first_img)
cv2.imwrite('uzh_vis_patch.jpg', patch)




#path = 'compressed'
#for i in os.listdir(os.path.join(os.getcwd(), path)):
#    #print(i)
#    image_path = os.path.join(os.getcwd(), path, i)
#    image = cv2.imread(image_path)
#
#    cv2.imwrite('res/{}.jpg'.format(i.split('.')[0]), image)

path = 'res'

for i in os.listdir(path):
    print(i)
#image = cv2.imread(path)
    image = cv2.imread(os.path.join('res', i))
    if image.shape[0] < 1000:
        image = cv2.resize(image, (256, 256))
    else:
        image = cv2.resize(image, (512, 512))


    print(image.shape)
    cv2.imwrite(os.path.join('res', i), image)

#image = cv2.resize(image, (1024, 1024))