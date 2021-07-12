import math
import os
import glob
import csv

import cv2
import numpy as np
import image_slicer



def slice_image(image, win_size):
    assert image.shape[0] > win_size and image.shape[1] > win_size

    h, w = image.shape[:2]
    print(image.shape)
    assert w %  win_size== 0
    assert h %  win_size== 0

    h_stride_num = h / win_size
    w_stride_num = w / win_size
    images = []
    #print(h, w, win_size)
    for h_idx in range(int(h_stride_num)):
        for w_idx in range(int(w_stride_num)):
            h_coord = h_idx * win_size
            w_coord = w_idx * win_size
            images.append({
                'img' : image[h_coord : h_coord+win_size, w_coord : w_coord+win_size],
                'h' : h_coord,
                'w' : w_coord
            })

            #print(h_coord, w_coord)
    return images

def read_csv(csv_fp):
    res = {}
    with open(csv_fp) as csv_file:
        spam_reader = csv.reader(csv_file)
        for row in spam_reader:
            #print(','.join(row))
            res[row[0]] = row[1]

    #print(res)
    return res

def extract_class_name(image_name):
    class_info ={
        'Normal' : '1_normal',
        'Low-Grade': '2_low_grade',
        'High-Grade': '3_high_grade',
        'Grade2': '2_low_grade',
        'Grade1': '1_normal',
        'Grade3': '3_high_grade',
    }

    for key, value in class_info.items():
        if key in image_name:
            return value

def save_patches(image_patches, save_path, src_image_name, fold_idx, printable=False):
    #if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

    #count = 0
    for patch in image_patches:
        image = patch['img']
        row_idx = patch['h']
        col_idx = patch['w']

        image_name = src_image_name.replace('.png', '_grade_{}_row_{:04}_col_{:04}.png'.format(int(fold_idx) + 1, row_idx, col_idx))
        patch_fp = os.path.join(save_path, image_name)

        if printable:
            print('saving files to {}....'.format(patch_fp))

        cv2.imwrite(patch_fp, image)

        #if count == 1:
            #import sys;sys.exit()

if __name__ == '__main__':
    #path = '/data/smb/数据集/结直肠/病理学/Extended_CRC/train/Grade2_Patient_007_029810_066073.png'
    src_path = '/data/smb/数据集/结直肠/病理学/Extended_CRC/Original_Images/'
    #dest_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_su/raw/Extended_CRC'
    dest_path = 'test_can_be_del2'

    csv_file = '/data/by/tmp/HGIN/dataflow/extended_crc_fold_info.csv'
    split_info = read_csv(csv_file)
    for i in glob.iglob(os.path.join(src_path, '**', '*.png'), recursive=True):
        #print(i)
        #a = image_slicer.slice(i, 32 * 24, save=False)
        #c = 0
        #for a1 in a:
        #    c += 1
        #    a1.save(
        #        'test_can_be_del/{}.png'.format(c)
        #    )
        #import sys; sys.exit()
        #print(a[0], type(a[0]))
        #a[0].save()
        #print(i)
        image = cv2.imread(i, -1)
        #patches = crop_image(image, 224)
        patches = slice_image(image, 224)
        image_name = os.path.basename(i)
        fold_id = split_info[image_name]
        class_name = extract_class_name(image_name)
        fold_folder = 'fold_{}'.format(int(fold_id) + 1)
        patch_save_path = os.path.join(dest_path, fold_folder, class_name)
        save_patches(patches, patch_save_path, image_name, fold_id, printable=False)

        #import sys;sys.exit()
        #print(len(patches))
        #print(patches[45]['img'].shape)
        #print(patches[45]['h'])
        #print(patches[45]['w'])

        #import sys;sys.exit()
    #image = cv2.imread(path, -1)
    #print(image.shape)
    #images = crop_image(image, 1792, 224)
    #print(len(images))
    #print(images[33]['img'].shape)