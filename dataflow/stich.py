import glob
import os
import re
import bisect

import cv2
import numpy as np
#from matplotlib import plot as plt
import matplotlib.pyplot as plt



class ImageLists:
    def image_list(self):
        NotImplementedError

    def read_image(self, image_path):
        NotImplementedError


class ImageFolder(ImageLists):
    def __init__(self, path):
        search_path = os.path.join(path, '**', '*.png')
        self.image_names = []
        for i in glob.iglob(search_path, recursive=True):
            self.image_names.append(i)

    def image_list(self):
        return self.image_names

    #def read_image(self, image_path):
    #    image = cv2.imread(image_path, -1)
    #    return image

class ImageLMDB(ImageLists):
    pass

class BaseDataset:
    def __init__(self, image_list, image_size):
        self.image_names224 = image_list.image_list()
        self.image_grids = self.construct_grids()
        self.image_size = image_size
        self.imagefp2coords = self.image_coords()
        assert image_size % 224 == 0
        self.border = int(image_size / 224)
        #self.cum_sums = self.cal_cum_sum()
        #print(self.cum_sums)
        #self.
        #print(self.image_grids.shape)
        #self.image_clusters = self.cluster_images()
        #print(image_clusters)
        #print(len(image_clusters))

    def image_coords(self):
        res = {}
        for k, v in self.image_grids.items():
            row, col = v.shape
            for r_idx in range(row):
                for c_idx in range(col):
                    res[v[r_idx, c_idx]] = (r_idx, c_idx)

        return res

    @property
    def image_names(self):
        res = []
        for k, v in self.image_grids.items():
            v = v[:-self.border + 1, :-self.border + 1]
            print(v.shape)
            res.append(v.flatten())

        return np.hstack(res)

    def stich_images(self, patch):
        row, col = patch.shape
        h = []
        v = []
        for r_idx in range(row):
            for c_idx in range(col):
                path = patch[r_idx, c_idx]
                image = cv2.imread(path, -1)
                h.append(image)
            image = np.hstack(h)
            #print(image.shape)
            v.append(image)
            h = []
        image = np.vstack(v)
        return image
            #cv2.hstack
    def grid_idx(self, image_grid, sample_idx):
        row, col = image_grid.shape
        row = row - self.border + 1
        col = col - self.border + 1
        #print(row, col)
        #print('sample_idx', sample_idx, 'row', row, 'col', col, 111)
        r_idx = int(sample_idx / col)
        c_idx = int(sample_idx % col)
        return r_idx, c_idx

    def get_image_by_path(self, path):
        r_idx, c_idx = self.imagefp2coords[path]
        patch = image_grid[r_idx : r_idx+self.border, c_idx : c_idx+self.border]
        image = self.stich_images(patch)
        return path, image

    #def path2laIbel(self, path):



    def __getitem__(self, idx):
        image_idx = bisect.bisect_right(self.cum_sum, idx)
        #print(image_idx)
        prefix = self.image_prefixes[image_idx]
        image_grid = self.image_grids[prefix]

        if image_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cum_sum[image_idx - 1]

        #print(sample_idx)
        #print(image_grid)
        r_idx, c_idx = self.grid_idx(image_grid, sample_idx)
        #print(r_idx, c_idx)
        patch = image_grid[r_idx : r_idx+self.border, c_idx : c_idx+self.border]
        path = image_grid[r_idx, c_idx]
        image = self.stich_images(patch)
        assert image.shape[0] == 1792
        assert image.shape[1] == 1792

        return path, image

    def __len__(self):
        return self.cum_sum[-1]
        #length = 0
        ##print(self.image_grids['test_can_be_del2/fold_3/2_low_grade/Grade2_Patient_002_073780_036698'])
        #for k, v in self.image_grids.items():
        #    v1, v2 = v.shape
        #    length += (v1 - self.border + 1) * (v2 - self.border + 1)
        #return int(length)

    def cal_seq_len(self, image):
        s1, s2 = image.shape
        return int((s1 - self.border + 1) * (s2 - self.border + 1))

    @property
    def image_prefixes(self):
        res = []
        #print(11, len(self.image_grids.keys()))
        for k, v in self.image_grids.items():
            res.append(k)
        return  res

    @property
    def cum_sum(self):
        res = []
        s = 0
        for k, v in self.image_grids.items():
            #v1, v2 = v.shape
            length = self.cal_seq_len(v)
            res.append(length + s)
            s += length
        return res

    def cluster_images(self):
        image_prefix = dict()
        for image_name224 in self.image_names224:
            prefix = image_name224.split('_grade_')[0]
            if prefix not in image_prefix:
                image_prefix[prefix] = []
            image_prefix[prefix].append(image_name224)

            #if 'Grade3_Patient_172_7_grade_3_row_4256' in image_name224:
            #    print(prefix, image_name224, image_prefix[prefix][-1])

        return image_prefix

    def construct_grids(self):
        image_clusters = self.cluster_images()
        #print(image_clusters)

        def row_col(path):
            base_name = os.path.basename(path)
            row, col = re.search(r'_row_([0-9]+)_col_([0-9]+)', path).groups()
            #print(int(row), int(col))
            return int(row), int(col)
            #return int(row)

        for k, v in image_clusters.items():
            #print(k, len(v))

            image_grids = []
            v = sorted(v, key=row_col)
            last_row = 0
            row = []
            for elem in v:
                r_idx, _ = row_col(elem)
                if last_row != r_idx:
                    image_grids.append(np.array(row))
                    last_row = r_idx
                    row = []

                row.append(elem)
                #if 'Grade3_Patient_172_7_grade_3_row_4256' in elem:
                #    print(elem, row[-1])

            image_grids.append(np.array(row))
            image_clusters[k] = np.array(image_grids)
            #print(image_clusters[k].shape)
            #print(image_clusters[k][3:5, 5:8])
            #print(image_clusters[k].shape)
            #print(image_clusters[k])

        return image_clusters


    def vis_image(self, image_idx, ori_folder, save_folder):
        for idx, (k, v) in enumerate(self.image_grids.items()):

            base_name = os.path.basename(k + '.png')
            #print(os.path.join(ori_folder, base_name))
            src_image = cv2.imread(os.path.join(ori_folder, base_name), -1)

            stich_image = self.stich_images(v)
            #print(v.shape)
            #print(v[-1])

            #print(stich_image.shape, src_image.shape)
            assert stich_image.shape == src_image.shape

            image = np.hstack([stich_image, src_image])
            image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)

            cv2.imwrite(os.path.join(save_folder, base_name), image)


            #for i in v:
                #print(i)
                #print(i)
                #pass

        #print(len(image_clusters.keys()))
        #print(type(image_clusters.values()))



image_folder = ImageFolder('test_can_be_del2/')
dataset = BaseDataset(image_folder, 1792)
print(len(dataset))

for i in range(300):
    dataset.vis_image(i, '/data/smb/数据集/结直肠/病理学/Extended_CRC/Original_Images/', 'test_can_be_del')
#for idx, image in enumerate(dataset):
    #pass
#image = dataset[2200]
#
#cv2.imwrite('test.png', image)
#
#image_names = dataset.image_names
#print(len(image_names))

    #print(image.shape)
    #cv2.imwrite('test_can_be_del/test{}.png'.format(idx), image)
