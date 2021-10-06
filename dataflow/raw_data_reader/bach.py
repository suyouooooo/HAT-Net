import csv
import glob
import os
import json

import numpy as np
import cv2

class RawDataReaderBACH:
    def __init__(self, image_folder, json_folder, dataset, return_mask=False):
        self.image2path = {}
        self.json2path = {}
        search_path = os.path.join(image_folder, '**', '*.tif')
        #search_path = os.path.join(image_folder, '**', '*.png')
        for image_fp in glob.iglob(search_path, recursive=True):
            image_name = os.path.basename(image_fp)
            self.image2path[image_name] = image_fp

        search_path = os.path.join(json_folder, '**', '*.json')
        for json_fp in glob.iglob(search_path, recursive=True):
            json_name = os.path.basename(json_fp)
            self.json2path[json_name] = json_fp

        self.image_names = list(self.image2path.keys())

        self.return_mask = return_mask
        self.dataset = dataset
        self.scale = 1

    def __len__(self):
        return len(self.image2path.keys())

    def read_file(self, json_path):
        res = []
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            for k, v in json_data['nuc'].items():
                res.append(v)
        return res

    def format_labels(self, labels, image_size):
        coords = []
        bboxes = []
        if self.return_mask:
            image_size = [s * self.scale for s in image_size]
            mask = np.zeros(image_size, dtype='uint8')

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
                cv2.drawContours(mask, [np.array(node['contour'])], -1, 255, -1)

        if self.return_mask:
            return bboxes, coords, mask
        else:
            return bboxes, coords, None

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_fp = self.image2path[image_name]
        image = cv2.imread(image_fp)
        image_size = image.shape[:2]

        #json_fp = self.json2path[image_name.replace('.png', '.json')]
        #json_fp = self.json2path[image_name.replace('.jpg', '.json')]
        json_fp = self.json2path[image_name.replace('.tif', '.json')]

        labels = self.read_file(json_fp)
        bboxes, coords, mask = self.format_labels(labels, image_size)

        image_name = os.path.basename(image_fp)

        if self.return_mask:
            mask = np.expand_dims(mask, axis=-1)
            assert image.shape[:2] == mask.shape[:2]

        return self.dataset(
                rel_path=image_name,
                image=image,
                bboxes=bboxes,
                coords=coords,
                mask=mask
                )

class RawDataReaderBACHTestSet:
    def __init__(self, image_folder, json_folder, dataset, csv_file, return_mask=False):
        self.image2path = {}
        self.json2path = {}
        search_path = os.path.join(image_folder, '**', '*.tif')
        #search_path = os.path.join(image_folder, '**', '*.png')
        for image_fp in glob.iglob(search_path, recursive=True):
            image_name = os.path.basename(image_fp)
            self.image2path[image_name] = image_fp

        search_path = os.path.join(json_folder, '**', '*.json')
        for json_fp in glob.iglob(search_path, recursive=True):
            json_name = os.path.basename(json_fp)
            self.json2path[json_name] = json_fp

        self.image_names = list(self.image2path.keys())

        self.return_mask = return_mask
        self.dataset = dataset
        self.scale = 1
        self.image_labels = self.read_csv(csv_file)

    def read_csv(self, csv_fp):
        image_names = []
        labels = []
        res = {}
        with open(csv_fp) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                case, label = row['case'], row['class']
                image_name = 'test{}.tif'.format(case)
                label = int(label) + 1
                if label == 1:
                    label = 2
                elif label == 2:
                    label = 1
                res[image_name] = label

        return res

    def __len__(self):
        return len(self.image2path.keys())

    def read_file(self, json_path):
        res = []
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            for k, v in json_data['nuc'].items():
                res.append(v)
        return res

    def format_labels(self, labels, image_size):
        coords = []
        bboxes = []
        if self.return_mask:
            #mask = np.zeros((self.image_size * self.scale, self.image_size * self.scale), dtype='uint8')
            image_size = [s * self.scale for s in image_size]

            mask = np.zeros(image_size, dtype='uint8')

        type_probs = []
        for node in labels:
            cen = node['centroid']
            cen = [int(c) for c in cen] # x, y
            cen = cen[::-1] # convert x y to y x
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
                cv2.drawContours(mask, [np.array(node['contour'])], -1, 255, -1)

        if self.return_mask:
            return bboxes, coords, mask
        else:
            return bboxes, coords, None

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_fp = self.image2path[image_name]
        image = cv2.imread(image_fp)
        #print(image.shape, 1111111111111)
        image_size = image.shape[:2]

        #json_fp = self.json2path[image_name.replace('.png', '.json')]
        #json_fp = self.json2path[image_name.replace('.jpg', '.json')]
        json_fp = self.json2path[image_name.replace('.tif', '.json')]

        labels = self.read_file(json_fp)
        bboxes, coords, mask = self.format_labels(labels, image_size)

        image_name = os.path.basename(image_fp)
        image_label = self.image_labels[image_name]
        image_name = image_name.replace('.', '_grade_{}.'.format(image_label))
        #print(image_name)
        #import sys;sys.exit()

        if self.return_mask:
            mask = np.expand_dims(mask, axis=-1)
            #print(image.shape, mask.shape)
            assert image.shape[:2] == mask.shape[:2]

        return self.dataset(
                rel_path=image_name,
                image=image,
                bboxes=bboxes,
                coords=coords,
                mask=mask
                )