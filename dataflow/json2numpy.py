import os
import glob
import json

import cv2
import numpy as np
import torch


class Patches:
    def __init__(self, image, mask, json_data, patch_size):
        """
            bboxes :bbox [min_y, min_x, max_y, max_x]
            coords: (row, col)
        """

        self.image = image
        self.mask = mask
        self.patch_size = patch_size
        self.inst_ids = np.unique(mask[:, :, 0])
        self.inst_ids = self.inst_ids[self.inst_ids != 0]
        self.json_data = json_data['nuc']

        #self.filter_out_low_probs()
        #from model.cpc import network
        #mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr
        #std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        #mean = [0.73646324, 0.56556627, 0.70180897] # Expanuke bgr
        #self.std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        #std = [0.18869222, 0.21968669, 0.17277594] # Expanuke bgr

    def filter_out_low_probs(self):
        res = []
        for inst_id in self.inst_ids:
            node = self.json_data[str(inst_id)]
            prob = node['type_prob']
            if prob < 0.5:
                continue
            res.append(inst_id)

        self.inst_ids = res

    #def entory(self, gray_image, d):
    #    return Entropy(gray_image, d)
    def __len__(self):
        return len(self.inst_ids)

    def __getitem__(self, idx):
        #bbox = self.bboxes[idx]
        #print(bbox, idx)
        #bbox = self.pad_patch(*bbox, self.patch_size)
        #t1 = time.time()
        inst_id = self.inst_ids[idx]
        inst_map = self.mask[:, :, 0]
        type_map = self.mask[:, :, 1]

        #t2 = time.time()
        #row_idx, col_idx = np.nonzero(inst_map == inst_id)
        #bbox = [row_idx.min(), col_idx.min(), row_idx.max(), col_idx.max()]

        node = self.json_data[str(inst_id)]
        type_id = node['type']
        bbox = node['bbox']
        bbox = [b for b in sum(bbox, [])]
        #print(bbox)

        #t3 = time.time()
        #max_type = type_map[inst_map == inst_id].max()
        #min_type = type_map[inst_map == inst_id].min()

        #assert max_type == min_type
        type_id = node['type']

        #t4 = time.time()
        #print('before', bbox)
        bbox = self.pad_patch(*bbox, self.patch_size)
        #image = self.image.copy()
        patch = self.image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1].copy()
        inst_map = inst_map[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
        type_map = type_map[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]

        #t5 = time.time()
        # hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
        mask = np.bitwise_or(type_map == type_id, type_map == 0)
        #mask = type_map == type_id
        #patch[mask] = 0
        #mask = np.bitwise_or(type_map == type_id, type_map == 0)
        patch[~mask] = 0
        # hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
        #patch = image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]

        #t6 = time.time()
        #top = max(int((patch.shape[0] - self.patch_size) / 2), 0)
        #bottom = max(0, top - patch.shape[0])
        #right = max(int((patch.shape[1] - self.patch_size) / 2), 0)
        #left = max(0, right - patch.shape[1])
        ##print(top, bottom, left, right)
        ##print('before pad', patch.shape)
        #patch = cv2.copyMakeBorder(patch, top, bottom, left, right, cv2.BORDER_REFLECT)
        #total = t6 - t1
        #print('{} {} {} {} {}'.format(
        #    (t2 - t1) / total,
        #    (t3 - t2) / total,
        #    (t4 - t3) / total,
        #    (t5 - t4) / total,
        #    (t6 - t5) / total,
        #    #t6 / total,
        #))

        #image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (255, 0, 0), thickness=1)

        #if patch.shape[0] != 65:
        #    print(patch.shape, bbox, idx, inst_id, type_id)
        #print(patch.shape)
        #print('total', total)
        #print(patch.shape)
        return patch, type_id
        #print('after pad', patch.shape)
        #print(image.shape)
        #patch = self.transforms(patch)
        #print(patch.shape)

        #print(patch.shape)
        #return patch, hand_features_and_coord
        #patch = self.transforms(patch)
        #return patch, np.array(self.coords[idx])

    def pad_patch(self, min_x, min_y, max_x, max_y, output_size):
    #def pad_patch(self, min_y, min_x, max_y, max_x, output_size):
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


def json2image(json_fp, output_size):
    #type_mask = np.zeros((output_size, output_size), dtype='int16')
    #inst_mask = np.zeros((output_size, output_size), dtype='int16')
    type_mask = np.zeros(output_size, dtype='int16')
    inst_mask = np.zeros(output_size, dtype='int16')

    json_data = json.load(open(json_fp))
    for node_id, node in json_data['nuc'].items():
        #print(k, type(k))
        #inst_mask[]
        #if int(node_id) == 0:
        #print(node_id)

        #print(node['type'])
        #print(type(node['type']))
        cv2.drawContours(type_mask, [np.array(node['contour'])], -1, node['type'], -1)
        cv2.drawContours(inst_mask, [np.array(node['contour'])], -1, int(node_id), -1)
        #cv2.drawContours(type_mask, contours=[np.array(node['contour'])], thickness=3, lineType=8, color=node['type'], contourIdx=-1)
        #cv2.drawContours(inst_mask, contours=[np.array(node['contour'])], thickness=3, lineType=8, color=int(node_id), contourIdx=-1)

    mask = np.dstack([inst_mask, type_mask])

    return mask
    #print(mask.shape)
    #print(np.unique(inst_mask))
    #print(np.unique(type_mask))
    #print(np.unique())

def gen_image_json_path(image_path, json_path):
    #for i in glob.iglob(os.path.join(json_path, '**', '*.json'), recursive=True):
    #for i in glob.iglob(os.path.join(image_path, '**', '*.jpg'), recursive=True):
    #for i in glob.iglob(os.path.join(image_path, '**', '*.tif'), recursive=True):
    for i in glob.iglob(os.path.join(image_path, '**', '*.png'), recursive=True):
        #base_name = os.path.basename(i).replace('.jpg', '.json')
        base_name = os.path.basename(i).replace('.png', '.json')
        json_fp = os.path.join(json_path, base_name)
        yield  i, json_fp

def draw_nuclei(image, mask):
    image = image.copy()
    inst_map = mask[:, :, 0]
    type_map = mask[:, :, 1]

    image[type_map != 0] = 0
    image[type_map == 1] = (0, 0, 255)
    image[type_map == 2] = (255, 0, 0)
    image[type_map == 3] = (0, 255, 0)
    image[type_map == 4] = (255, 255, 0)
    image[type_map == 5] = (255, 0, 255)

    return image

class WriterParallel:
    def __init__(self, num_threads, save_path):
        self.thread_pool = pool.ThreadPool(num_threads)
        self.save_path = save_path

    def write_file(self, image, label, count):
        base_name = 'img_{}_{}.jpg'.format(count, label)
        save_path = os.path.join(self.save_path, base_name)
        cv2.imwrite(save_path, image)

    #def write(self, images, labels, counts):
    def write(self, args):
        self.thread_pool.map(self.write_file, args)


def write_lmdb(image_path, lmdb_path):
    map_size = 10 << 40
    #print(lmdb_path)
    env = lmdb.open(lmdb_path, map_size=map_size)

    map_size = 10 << 40

    with env.begin(write=True) as txn:
        for image_fp in glob.iglob(os.path.join(image_path, '**', '*.jpg'), recursive=True):
            #image = cv2.imread(image_fp, -1)
            #image = pickle.dumps(image)
            image = open(image_fp, 'rb').read()
            image_name = os.path.basename(image_fp)
            txn.put(image_name.encode(), image)


#json_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/json_withtypes/json/'
#json_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Json/withtypes/train/json/'
#json_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Json/withtypes/test/json/'
json_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Json/SliceJson/json'
#image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/images/'
#image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Images/test'
image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/SliceImage/'
#save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/Patches_Binary'
save_path = '/data/hdd1/by/HGIN/tmp1'
#save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/NumpyMask'
#save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/Patches'
#save_path = '/data/hdd1/by/dataset/Patches'
#save_path = '/data/hdd1/by/dataset/tmp'
#save_path = '/data/hdd1/by/HGIN/tttt_del'

datasets = []
count = 0
import random
#cc = random.choice(range(886))
import time
start = time.time()
for idx, (image_fp, json_fp) in enumerate(gen_image_json_path(image_path, json_path)):
    #print(i)
    #if count != cc:
    #    continue

    #print(cc)
    #if idx != 36:
    #    continue
    print(image_fp)
    print(json_fp)
    image = cv2.imread(image_fp)
    #print(image.shape)
    mask = json2image(json_fp, image.shape[:2])
    json_data = json.load(open(json_fp))

    #image = draw_nuclei(image, mask)
    #cv2.imwrite('jj.jpg', image)

    #import sys; sys.exit()

    dataset = Patches(image, mask, json_data, 64)
    t1 = time.time()
    c1 = 0
    c2 = 0
    patch = dataset

    for image, label in dataset:
        count += 1
        #t2 = time.time()
        #c = t2 - t1
        #print('computing.....: {:02f}'.format(t2 - t1))
        #c1 += t2 - t1
        save_fp = os.path.join(save_path, 'img_{}_{}.jpg'.format(count, label))
        print(save_fp)
        #cv2.imwrite(os.path.join(save_path, 'img_{}_{}.jpg'.format(count, label)), image)
        #t1 = time.time()
        #w = t1 - t2
        #print('writting....: {:02f}'.format(t1 - t2))
        #print(w / c)
        #c2 += t1 - t2
        #print(c1 / c2)


    print('avg: {:02f}, total: {:02f}'.format((idx + 1) / (time.time() - start), time.time() - start))

    print(idx)

print(count)
    #image = draw_nuclei(image, mask)
    #dataset = Patches(image, mask, 64)
    #patch, label, image = dataset[33]
    #cv2.imwrite('ddd.jpg', patch)
    #print(label)

    #if count == 10:
    #    break





    #image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
    #cv2.imwrite('test1.jpg', image)
    #import sys;sys.exit()

    #if count == 5:
    #    break
    #print(image_fp, json_fp)

    #base_name = os.path.basename(i).replace('.json', '.npy')
    #break
    #np_fp = os.path.join(save_path, base_name)

#datasets = torch.utils.data.ConcatDataset(datasets)
#data_loader = torch.utils.data.DataLoader(datasets, num_workers=4, batch_size=64 * 4, shuffle=False)
#print('------------------------')
#count = 0
#for idx, (images, labels) in enumerate(data_loader):
#    print('fff')
#    for image, label in zip(images, labels):
#        count += 1
#        print(image, type(image))
#        cv2.imwrite(os.path.join('img_{}_{}.jpg'.format(count, label)), image.numpy())
#
#    print('[{}]/[{}]'.format(idx, len(datasets)))
    #    image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (255, 0, 0), thickness=1)

#print(len(datasets))
#count = 0
#for i in datasets:
#    count += 1
#
#    cv2.imwrite(os.path.join(save_path, 'img_{}.jpg'.format(count)), i)



    #np.save(np_fp, mask)
    #print(np_fp)