import glob
import os
import re
import bisect
import json
import string
import pickle
from pathlib import Path

import cv2
import numpy as np
import lmdb
import torch


#class BaseLists:
#    def file_list(self):
        #return


class BaseLists:
    def __init__(self, path):
        self.path = path

    def file_list(self):
        NotImplementedError

    def read_file(self, image_path):
        NotImplementedError

class PtFolders(BaseLists):
    def __init__(self, path):
        super().__init__(path)
        search_path = os.path.join(path, '**', '*.pt')
        self.pt_names = []
        for i in glob.iglob(search_path, recursive=True):
            self.pt_names.append(i)

    def file_list(self):
        return self.pt_names

    def read_file(self, pt_path):
        return torch.load(pt_path)

class JsonFolder(BaseLists):
    def __init__(self, path):
        super().__init__(path=path)
        search_path = os.path.join(path, '**', '*.json')
        self.json_names = []
        for i in glob.iglob(search_path, recursive=True):
            self.json_names.append(i)

    def file_list(self):
        return self.json_names

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

class ImageFolder(BaseLists):
    def __init__(self, path):
        super().__init__(path=path)
        search_path = os.path.join(path, '**', '*.png')
        self.image_names = []
        for i in glob.iglob(search_path, recursive=True):
            self.image_names.append(i)

    def file_list(self):
        return self.image_names

    def read_file(self, image_path):
        image = cv2.imread(image_path, -1)
        return image

    #def image_list(self):
    #    return self.image_list()
    #def read_image(self, image_path):
    #    image = cv2.imread(image_path, -1)
    #    return image

class LMDBFolder(BaseLists):
    def __init__(self, path):
        self.env = lmdb.open(path, map_size=1099511627776, readonly=True, lock=False)
        #with self.env.begin(write=False) as txn:
        #    self.image_names= [key.decode() for key in txn.cursor().iternext(keys=True, values=False)]

        cache_file = '_cache_' + ''.join(c for c in path if c in string.ascii_letters)
        cache_path = os.path.join(path, cache_file)
        if os.path.isfile(cache_path):
            self.npy_names = pickle.load(open(cache_path, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.npy_names = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.npy_names , open(cache_path, "wb"))

        self.npy_names = [key.decode() for key in self.npy_names]

    def file_list(self):
        return self.npy_names

    def read_file(self, npy_path):
        #image = cv2.imread(image_path, -1)
        with self.env.begin(write=False) as txn:
            data = txn.get(npy_path.encode())
            data = pickle.loads(data)
        return data

class BaseDataset:
    def __init__(self, file_list, image_size):
        self.file_names = file_list.file_list()
        self.file_grids = self.construct_grids()
        self.image_size = image_size
        self.imagefp2coords = self.image_coords()

        assert image_size % 224 == 0
        self.border = int(image_size / 224)
        self.file_list = file_list
        #self.cum_sums = self.cal_cum_sum()
        #print(self.cum_sums)
        #self.
        #print(self.file_grids.shape)
        #self.image_clusters = self.cluster_files()
        #print(image_clusters)
        #print(len(image_clusters))

    #abstrct
    def image_coords(self):
        res = {}
        for k, v in self.file_grids.items():
            #print(v.shape, v)
            row, col = v.shape
            for r_idx in range(row):
                for c_idx in range(col):
                    res[v[r_idx, c_idx]] = (r_idx, c_idx)

        return res

    #abstrct
    @property
    #def image_path_lists(self):
    def file_path_lists(self):
        res = []
        for k, v in self.file_grids.items():
            if self.border != 1:
                v = v[:-self.border + 1, :-self.border + 1]
            res.append(v.flatten())

        return np.hstack(res)

    # not abstrct
    def stich_files(self, files):
        NotImplementedError
    #def stich_files(self, patch):
    #    row, col = patch.shape
    #    h = []
    #    v = []
    #    for r_idx in range(row):
    #        for c_idx in range(col):
    #            path = patch[r_idx, c_idx]
    #            image = cv2.imread(path, -1)
    #            h.append(image)
    #        image = np.hstack(h)
    #        #print(image.shape)
    #        v.append(image)
    #        h = []
    #    image = np.vstack(v)
    #    return image
            #cv2.hstack

    #abstrct
    def grid_idx(self, file_grid, sample_idx):
        row, col = file_grid.shape
        row = row - self.border + 1
        col = col - self.border + 1
        r_idx = int(sample_idx / col)
        c_idx = int(sample_idx % col)
        return r_idx, c_idx

    #abstrct
    def get_file_by_path(self, path):
        #if path not in self.imagefp2coords:
        #    r_idx, c_idx = self.imagefp2coords[os.path.basename(path)]
        #else:
        #print(self.imagefp2coords.keys())
        # print(self.imagefp2coords.keys()[3])
        r_idx, c_idx = self.imagefp2coords[path]
        base_name = os.path.basename(path)
        #print(base_name)
        image_prefix = base_name.split('_grade_')[0]
        file_grid = self.file_grids[image_prefix]
        patch = file_grid[r_idx : r_idx+self.border, c_idx : c_idx+self.border]
        #print(patch, self.stich_files)
        data = self.stich_files(patch)
        #print(data)
        return data

    #abstrct
    def assert_data(data):
        NotImplementedError

    #abstrct
    def __getitem__(self, idx):
        image_idx = bisect.bisect_right(self.cum_sum, idx)
        prefix = self.file_prefixes[image_idx]
        file_grid = self.file_grids[prefix]

        if image_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cum_sum[image_idx - 1]

        r_idx, c_idx = self.grid_idx(file_grid, sample_idx)
        patch = file_grid[r_idx : r_idx+self.border, c_idx : c_idx+self.border]
        path = file_grid[r_idx, c_idx]
        #image = self.stich_files(patch)
        output = self.stich_files(patch)

        self.assert_data(output)
        #assert image.shape[0] == self.image_size
        #assert image.shape[1] == self.image_size

        #return path, image
        return path, output

    #abstrct
    def __len__(self):
        return self.cum_sum[-1]
        #length = 0
        #for k, v in self.file_grids.items():
        #    v1, v2 = v.shape
        #    length += (v1 - self.border + 1) * (v2 - self.border + 1)
        #return int(length)

    #abstrct
    def cal_seq_len(self, image):
        s1, s2 = image.shape
        return int((s1 - self.border + 1) * (s2 - self.border + 1))

    #abstrct
    @property
    def file_prefixes(self):
        res = []
        for k, v in self.file_grids.items():
            res.append(k)
        return  res

    #abstrct
    @property
    def cum_sum(self):
        res = []
        s = 0
        for k, v in self.file_grids.items():
            #v1, v2 = v.shape
            length = self.cal_seq_len(v)
            res.append(length + s)
            s += length
        return res

    #abstrct
    @property
    def cluster_files(self):
        file_prefix = dict()
        for file_name in self.file_names:
            base_name = os.path.basename(file_name)
            prefix = base_name.split('_grade_')[0]
            if prefix not in file_prefix:
                file_prefix[prefix] = []
            file_prefix[prefix].append(file_name)

        return file_prefix


    #abstrct
    def construct_grids(self):
        file_clusters = self.cluster_files
        #print(image_clusters)

        def row_col(path):
            base_name = os.path.basename(path)
            row, col = re.search(r'_row_([0-9]+)_col_([0-9]+)', path).groups()
            return int(row), int(col)

        for k, v in file_clusters.items():
            file_grids = []
            v = sorted(v, key=row_col)
            last_row = 0
            row = []
            for elem in v:
                r_idx, _ = row_col(elem)
                if last_row != r_idx:
                    file_grids.append(np.array(row))
                    last_row = r_idx
                    row = []

                row.append(elem)

            file_grids.append(np.array(row))
            file_clusters[k] = np.array(file_grids)

        return file_clusters

    def prefix2image_name(self, prefix):
        "full size"
        v = self.file_grids[prefix]
        return v[0, 0]

    def whole_file(self, prefix):
        #for idx, (k, v) in enumerate(self.file_grids.items()):
        prefix = str(prefix)
        v = self.file_grids[prefix]
        return self.stich_files(v)


class ImageDataset(BaseDataset):
    #def stich_files

    def assert_data(self, image):
        assert image.shape[0] == self.image_size
        assert image.shape[1] == self.image_size

    def stich_files(self, patch):
        row, col = patch.shape
        h = []
        v = []
        for r_idx in range(row):
            for c_idx in range(col):
                path = patch[r_idx, c_idx]
                #image = cv2.imread(path, -1)
                image = self.file_list.read_file(path)
                h.append(image)
            image = np.hstack(h)
            v.append(image)
            h = []
        image = np.vstack(v)
        return image

    #not abstrct
    def vis_image(self, ori_folder, save_folder):
        #ori_folder = self.file_list.path
        for idx, (k, v) in enumerate(self.file_grids.items()):

            base_name = os.path.basename(k + '.png')
            src_image = cv2.imread(os.path.join(ori_folder, base_name), -1)

            stich_image = self.stich_files(v)

            assert stich_image.shape == src_image.shape

            image = np.hstack([stich_image, src_image])
            image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)


            cv2.imwrite(os.path.join(save_folder, base_name), image)

class LMDBDataset(BaseDataset):
    def assert_data(self, val):
        assert 'feat' in val
        assert 'coord' in val

    def stich_files(self, patch):
        #res = []
        node_features = []
        node_coords = []

        #print(patch)
        #for fp in patch.flatten():
        row, col = patch.shape[:2]
        for r_idx in range(row):
            for c_idx in range(col):
                fp = patch[r_idx, c_idx]
                nodes = self.file_list.read_file(fp)
                #print(r_idx, c_idx, len(nodes))
                #for node in nodes:
                    # centeroid [x, y] # /data/by/tmp/hover_net/models/hovernet/post_proc.py  process
                    #node['centroid'][0] /= 2
                    #node['centroid'][1] /= 2
                    #print(node['centroid'])
                nodes['coord'][:, 0] += r_idx * 224 * 2
                nodes['coord'][:, 1] += c_idx * 224 * 2
                #node['bbox'][0][0]
                #node['bbox'][0][0] += r_idx * 224 * 2
                #node['bbox'][0][1] += c_idx * 224 * 2
                #node['bbox'][1][0] += r_idx * 224 * 2
                #node['bbox'][1][1] += c_idx * 224 * 2

                #print(node['contour'])
                #print(type(node['contour']))
                #node['contour'][:, 0] += r_idx * 224 * 2
                #node['contour'][:, 1] += c_idx * 224 * 2
                #print(node['contour'])
                #node['contour'] = [[c1 + c_idx * 224 * 2, c2 + r_idx * 224 * 2] for [c1, c2] in node['contour']]
                #print(node['contour'])

                #import sys; sys.exit()

                #node['centroid'][0] /= 2
                #node['centroid'][1] /= 2
                #print(node['centroid'])
                node_features.append(nodes['feat'])
                node_coords.append(nodes['coord'])
                #res.append(nodes)
        #    with open(fp) as f:
        #        data = json.load(fp)
        #        for v in data['nuc'].values():
        #            res.append(v)
        node_features = np.vstack(node_features)
        node_coords = np.vstack(node_coords)
        #print(res[33], 333333333)

        return {'feat' : node_features, 'coord' : node_coords}

class PtDataset(BaseDataset):
    def assert_data(self, data):
        assert 'feat' in data
        assert 'coord' in data

    def stich_files(self, patch, scale=2):
        res = {
            'feat' : [],
            'coord' : []
        }
        row, col = patch.shape[:2]
        for r_idx in range(row):
            for c_idx in range(col):
                fp = patch[r_idx, c_idx]
                data = self.file_list.read_file(fp)
                assert len(data['feat']) == len(data['coord'])
                feat = data['feat']
                coord = data['coord']
                if len(data['feat']) != 0:
                    coord[:, 1] += c_idx * 224 * scale
                    coord[:, 0] += r_idx * 224 * scale

                res['feat'].append(feat)
                res['coord'].append(coord)

        res['feat'] = torch.cat(res['feat'], dim=0)
        res['coord'] = torch.cat(res['coord'], dim=0)

        return res


class JsonDataset(BaseDataset):
    def assert_data(self, data):
        assert len(data) > 0
        assert type(data) == list

    def stich_files(self, patch, scale=2):
        res = []

        #for fp in patch.flatten():
        row, col = patch.shape[:2]
        for r_idx in range(row):
            for c_idx in range(col):
                fp = patch[r_idx, c_idx]
                nodes = self.file_list.read_file(fp)
                for node in nodes:
                    # centeroid [x, y] # /data/by/tmp/hover_net/models/hovernet/post_proc.py  process
                    #node['centroid'][0] /= 2
                    #node['centroid'][1] /= 2

                    # unit_size = 224 * 2
                    node['centroid'][0] += c_idx * 224 * scale
                    node['centroid'][1] += r_idx * 224 * scale
                    #node['bbox'][0][0]
                    node['bbox'][0][0] += r_idx * 224 * scale
                    node['bbox'][0][1] += c_idx * 224 * scale
                    node['bbox'][1][0] += r_idx * 224 * scale
                    node['bbox'][1][1] += c_idx * 224 * scale

                    #node['contour'][:, 0] += r_idx * 224 * 2
                    #node['contour'][:, 1] += c_idx * 224 * 2
                    #print(node['contour'])
                    # contour: (x,y)
                    node['contour'] = [[c1 + c_idx * 224 * scale, c2 + r_idx * 224 * scale] for [c1, c2] in node['contour']]
                    #print(node['contour'])

                    #import sys; sys.exit()

                    #node['centroid'][0] /= 2
                    #node['centroid'][1] /= 2
                    #print(node['centroid'])
                    res.append(node)
        #    with open(fp) as f:
        #        data = json.load(fp)
        #        for v in data['nuc'].values():
        #            res.append(v)
        #print(res[33], 333333333)

        return res
           #for i in v:
               #print(i)
               #print(i)
               #pass

       #print(len(image_clusters.keys()))
       #print(type(image_clusters.values()))

def draw_nuclei(image, json_label):
    #print(json_label)
    for node in json_label:
        #print(11, node)
        cen = node['centroid']
        image = cv2.circle(image, tuple(cen), 3, (0, 200, 0), cv2.FILLED, 1)
        # print(node['centroid'], node['bbox'])

        # cnt = [cnt // 2 for cnt in ]
        # node['contour'] = [[c1 // 2, c2 // 2] for [c1, c2] in contour]
        # cnt = [[c1 // 2, c2 // 2] for [c1, c2] in node['contour']]

        # image = cv2.drawContours(image, [np.array(cnt)], -1, 255, -1)


    return image

#image_folder = ImageFolder('/home/baiyu/Extended_CRC')
#image_dataset = ImageDataset(image_folder, 224 * 3)
#lmdb_folder = LMDBFolder('/data/smb/syh/PycharmProjects/CGC-Net/data_lmdb/extended_crc/feat/')
#feat_dataset = LMDBDataset(lmdb_folder, 224 * 3)
#print(len(image_dataset))
#print(len(feat_dataset))

#import random
#path, image = random.choice(image_dataset)
#
#print(path)
#base_name = os.path.basename(path).replace('.png', '.npy')
#lmdb_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_lmdb/extended_crc/feat/'
#
#image_path = Path(path)
#sub_folder = os.path.dirname(image_path.relative_to('/home/baiyu/Extended_CRC'))
#print(sub_folder)
#
#
#npy_path = os.path.join(sub_folder, base_name)
##print(npy_path)
#_, res = feat_dataset.get_file_by_path(npy_path)
#coords = res['coord']
#for cen in coords:
#    cen = [int(c // 2) for c in cen]
#    image = cv2.circle(image, tuple(cen[::-1]), 3, (0, 200, 0), cv2.FILLED, 3)
#
#cv2.imwrite('/home/baiyu/HGIN/heihei_del11.png', image)



#print(res)
#s = image_dataset.file_prefixes
#print(s[10])
#print(path)
#s = feat_dataset.file_prefixes
#print(s[10])
#s = feat_dataset.get_file_by_path
#image_dataset.vis_image('/data/smb/?????????/?????????/?????????/Extended_CRC/Original_Images/', 'tmp')
#sys.exit()
#
##json_folder = JsonFolder('/data/by/tmp/hover_net/samples/out/fold')
# json_folder = JsonFolder('/data/smb/syh/PycharmProjects/CGC-Net/data_su/raw/Extended_CRC/mask/fold_1/1_normal')
# print('ffffff')
# json_folder = JsonFolder('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Json/EXtended_CRC_Mask')
# print('fffffffffff')
# json_dataset = JsonDataset(json_folder, 1792)
# json_path, label = json_dataset[33]
# print(len(label), json_path)
#
#print(len(json_dataset))
#print(len(image_dataset))
#a = json_dataset[33]
#
#image_files = image_datasnt.file_path_lists
#json_files = json_dataset.file_path_lists
#
#print('json_files', len(json_files))
#print('image_files', len(image_files))
#json_path = json_files[0]
#
#json_dir = os.path.dirname(json_path)
#
#
#image_fp = image_files[1111]
#
#image_name = os.path.basename(image_fp)
#
#json_path = os.path.join(json_dir, image_name.replace('.png', '.json'))
#
##print(os.path.basename(json_path), os.path.basename(image_fp))
#print(image_fp)
#json_label = json_dataset.get_file_by_path(json_path)[1]
##print(type(json_label), len(json_label), json_label[0], 444444, json_path)
#image = image_dataset.get_file_by_path(image_fp)[1]
#
#
#print(len(json_label))
#image = draw_nuclei(image, json_label)
#print('ffffffffffffffffffffffffffffff')
##cv2.imwrite('test1_new.png', image)
#
#
#
#
#
#
#
#
##for i in range(300):
##    image_dataset.vis_image('/data/smb/?????????/?????????/?????????/Extended_CRC/Original_Images/', 'tmp')
#
#
##image_folder = ImageFolder('test_can_be_del2/')
##dataset = BaseDataset(image_folder, 224)
##print(len(dataset))
###
##for i in range(300):
##    dataset.vis_image(i, '/data/smb/?????????/?????????/?????????/Extended_CRC/Original_Images/', 'test_can_be_del')
##for idx, image in enumerate(dataset):
#    #pass
##image = dataset[2200]
##
##cv2.imwrite('test.png', image)
##
##image_names = dataset.image_names
##print(len(image_names))
#
#    #print(image.shape)
#    #cv2.imwrite('test_can_be_del/test{}.png'.format(idx), image)
#


#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat_hand_crafted/0/'
#
#
#
#folder = PtFolders(path)
#dataset = PtDataset(folder, 224 * 8)
#print(len(dataset))
#import random
#path, data = random.choice(dataset)
##data = dataset[33][1]
##path = dataset[33][0]
#print(path)
# image_folder = ImageFolder('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Images')
# img_dataset = ImageDataset(image_folder, 224 * 8)
# res = img_dataset[33]
# img_dataset.get_file_by_path()
# image_path = os.path.basename(json_path).replace('.json', '.png')
# image = img_dataset.get_file_by_path(os.path.join('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Json/EXtended_CRC_Mask', image_path))


# print(res[1].shape, res[0])

# image = res[1]
# image = cv2.resize(image, (0, 0), fx=2, fy=2)
# image = draw_nuclei(image, label)
# image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
# cv2.imwrite('aa.jpg', image)
#image_path = path.replace('Feat_hand_crafted', 'Images').replace('.pt', '.png').replace('0/', '')
##print(path)
##print(image_path)
#
# print(len(img_dataset), len(json_dataset))
##image = img_dataset[33][1]
#image = img_dataset.get_file_by_path(image_path)
#image = cv2.resize(image, (0, 0), fx=2, fy=2)
#
#print(type(data), len(data))
##image = cv2.imread(image_path)
#print(image.shape)
#coord = data['coord']
#for c in coord:
#        #print(c)
#        image = cv2.circle(image, tuple(c.tolist()[::-1]), 3, (330, 0, 0), 3)
#
#image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
#cv2.imwrite('fff.jpg', image)
##for k, v in data.items():
##    print(k, v.shape)
#
#
##for k in data['coord']:
##    print(k)
#
#
#
##path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Feat_hand_crafted/0/'
##image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Images/'
##
##count = 0
##for i in glob.iglob(os.path.join(path, '**', '*.pt'), recursive=True):
##    count += 1
##    if count != 10000:
##        continue
##    print(i)
##    dirname = os.path.dirname(i)
##    basename = os.path.basename(i)
##    dirnames = dirname.split('/')[-2:]
##    print(dirnames)
##    img_fp = os.path.join(image_path, *dirnames, basename.replace('.pt', '.png'))
##    print(img_fp)
##    image = cv2.imread(img_fp)
##
##    image = cv2.resize(image, (0, 0), fx=2, fy=2)
##    data = torch.load(i)
##    coord = data['coord']
##    #print(coord.shape)
##    #print(coord[:, 0].shape)
##    for c in coord:
##        #print(c)
##        image = cv2.circle(image, tuple(c.tolist()[::-1]), 3, (330, 0, 0), 3)
##
##    cv2.imwrite('fff.jpg', image)
##    break
#


#image_folder = ImageFolder('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Images/')
#dataset = ImageDataset(image_folder, 1792)
#
##print(len(dataset))
#image = dataset.get_file_by_path('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Images/fold_3/3_high_grade/Grade3_Patient_172_9_grade_3_row_2688_col_4256.png')
#print(image.shape)
#cv2.imwrite('ecrc.jpg', image)




if __name__ == '__main__':
    image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Images'
    json_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Json/EXtended_CRC_Mask'

    image_folder = ImageFolder(image_path)
    image_dataset = ImageDataset(image_folder, 224 * 8)
    image_path, image = image_dataset[44]


    json_folder = JsonFolder(json_path)
    json_dataset = JsonDataset(json_folder, 224 * 8)
    json_fp = os.path.join(json_path, os.path.basename(image_path).replace('.png', '.json'))
    json_label = json_dataset.get_file_by_path(json_fp)

    # image = cv2.resize(image, (0, 0), fx=2, fy=2)
    image = draw_nuclei(image, json_label)
    # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite('aa.jpg', image)
