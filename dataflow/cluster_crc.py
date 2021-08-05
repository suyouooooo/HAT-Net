import os
import re
import glob

import cv2
import numpy as np


def cluster(pathes):
    res = dict()

    for fp in pathes:
        base_name = os.path.basename(fp)
        prefix = base_name.split('_grade_')[0]

        if prefix not in res:
            res[prefix] = []

        res[prefix].append(fp)

    return res

def coords(path):
    base_name = os.path.join(path)
    row, col = re.search(r'_([0-9]+)_([0-9]+).png', base_name).groups()
    return int(row), int(col)
    #return int(col), int(row)


def stich(pathes):
    image = np.zeros((10000, 10000, 3), dtype='uint8')

    for fp in pathes:
        patch = cv2.imread(fp)
        row, col = coords(fp)
        h, w = patch.shape[:2]
        image[row : row+h, col : col+w] = patch

    h_idx, w_idx = np.nonzero(image)[:2]
    h_idx = np.max(h_idx)
    w_idx = np.max(w_idx)
    image = image[:h_idx + 1, :w_idx + 1]
    #import sys;sys.exit()
    print(image.shape)

    return image


src_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC/'
#
#
#split_image = ''
#
#
res = cluster(glob.iglob(os.path.join(src_path, '**', '*.png'), recursive=True))
#
count = 0
for k, v in res.items():
    count += 1
    print(k, len(v), 1111)
    print(v[0])
    print(count)
    print(coords(v[0]))
    image = stich(v)
    image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
    cv2.imwrite('testaa.png', image)

    import sys;sys.exit()
