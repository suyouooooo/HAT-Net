import os
import glob
import csv
from operator import itemgetter

import cv2
import numpy as np

import sys
sys.path.append(os.getcwd())
import dataflow
from dataflow.jsonreader import NucleiReader


image_path = '/data/smb/syh/pycharmprojects/cgc-net/data_prostate_tcga/data/images'
p1_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_prostate_tcga/data/labels/Gleason_masks_test_pathologist1/'
p2_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_prostate_tcga/data/labels/Gleason_masks_test_pathologist2/'
ptrain_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_prostate_tcga/data/labels/Gleason_masks_train'
search_path = os.path.join(image_path, '**', '*.jpg')

# BGR
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def gen_label(mask):
    #print(mask.shape)
    grade3_mask = np.all(mask==BLUE, axis=2)
    grade4_mask = np.all(mask==YELLOW, axis=2)
    #print(grade4_mask.shape)
    grade5_mask = np.all(mask==RED, axis=2)

    grade0_mask = np.all(mask==GREEN, axis=2)

    grade4_sum = grade4_mask.sum()
    grade3_sum = grade3_mask.sum()
    grade5_sum = grade5_mask.sum()
    grade0_sum = grade0_mask.sum()

    # blue, yellow, red
    res = [grade3_sum, grade4_sum, grade5_sum]
    #print(sum(res) + grade0_mask)
    #print(sum(res) + grade0_sum)
    if sum(res) + grade0_sum == 0:
        return -1, -1

    if sum(res) < 1550 * 1550 // 10:
        return -1, -1


    if sum(res) == 0 and grade0_sum != 0:
        return (0, 0), 0
    #res = sorted(res, reverse=True)
    #print(res)
    #print(sum(res), )

    indices, L_sorted = zip(*sorted(enumerate(res), key=itemgetter(1), reverse=True))
    #if sum(res) == 0:
        #print(res)
    #print(indices)
    #print(L_sorted)

    #print(indices)
    res = (indices[0] + 3, (indices[1] if L_sorted[1] else indices[0]) + 3)
    #print(res, sum(res))
    return res, sum(res)
    #tmp_mask1 = dilate(image, grade3_mask)
    #tmp_mask2 = dilate(image, grade4_mask)
    #tmp_mask3 = dilate(image, grade5_mask)
    ##tmp = dilate(tmp, grade3_mask)
    #image[tmp_mask1 != 0, :] = 0
    #image[tmp_mask2 != 0, :] = 0
    #image[tmp_mask3 != 0, :] = 0
    #tmp = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
    #cv2.imwrite('333333.jpg', tmp)

def dilate(image, mask):
    kernel = np.ones((5, 5), np.uint8)
    #print(mask.shape, 9999999)
    #print(mask.dtype)
    dal = cv2.dilate(mask.astype('uint8'), kernel, iterations=4)
    tmp_mask = np.bitwise_xor(mask, dal)
    #tmp = image.copy()
    #print(tmp_mask.shape, 11111111)
    #tmp[tmp_mask != 0, :] = 0
    return tmp_mask

def generate_csv_path(label_folder):
    norm_path = os.path.normpath(label_folder)
    csv_name = os.path.basename(norm_path) + '.csv'
    csv_fp = os.path.join(os.path.dirname(norm_path), csv_name)
    return csv_fp

def write_gleason_score(image_folder, label_folder):
    search_path = os.path.join(image_folder, '**', '*.jpg')
    img2path = {}
    for img_fp in glob.iglob(search_path, recursive=True):
        basename = os.path.basename(img_fp)
        img2path[basename] = img_fp


    search_path = os.path.join(label_folder, '**', '*.png')
    #count = 0
    header = ['image name', 'primary score', 'secondary score', 'gleason score']
    #csv_fp = generate_csv_path(label_folder)
    csv_fp = 'test.csv'
    #print(csv_fp)
    with open(csv_fp, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for mask_path in glob.iglob(search_path, recursive=True):
            mask_name = os.path.basename(mask_path)
            prefix = mask_name.split('_')[0]
            image_name = mask_name.replace(prefix + '_', '').replace('.png', '.jpg')
            image_path = img2path[image_name]
            image_name = os.path.basename(image_path)

            print(mask_path)
            mask = cv2.imread(mask_path, -1)
            grades, score = gen_label(mask)
            print(grades, score)
            #if not grades and not score:
            #print(grades, score)
            if grades == -1 and score == -1:
                print('kkkkkkkkkkkkkkkk')
                continue
            row = [image_name, *grades, score]
            writer.writerow(row)

#def write_gleason_score(image_folder, label_folder):
#
#    search_path = os.path.join(label_folder, '**', '*.png')
#    #count = 0
#    header = ['image name', 'primary score', 'secondary score', 'gleason score']
#    csv_fp = generate_csv_path(label_folder)
#    #print(csv_fp)
#    with open(csv_fp, 'w', encoding='UTF8') as f:
#        writer = csv.writer(f)
#        writer.writerow(header)
#        for mask_path in glob.iglob(search_path, recursive=True):
#            mask_name = os.path.basename(mask_path)
#            prefix = mask_name.split('_')[0]
#            image_name = mask_name.replace(prefix + '_', '').replace('.png', '.jpg')
#            image_path = img2path[image_name]
#            image_name = os.path.basename(image_path)
#
#            mask = cv2.imread(mask_path, -1)
#            grades, score = gen_label(mask)
#            if not grades and not score:
#                continue
#            row = [image_name, *grades, score]
#            writer.writerow(row)

            #image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
            #mask = cv2.resize(mask, (0, 0), fx=0.1, fy=0.1)
            #print(image_path)
            #print(mask_path)
            #cv2.imwrite('11111.jpg', image)
            #cv2.imwrite('22222.jpg', mask)

#write_gleason_score(image_path, p2_path)
#write_gleason_score(image_path, p1_path)
#write_gleason_score(image_path, ptrain_path)
#for i in glob.iglob(search_path, recursive=True):
#    print(os.path.basename(i).split('_')[0])
#    break
#
#search_path = os.path.join(ptrain_path, '**', '*.png')
#for i in glob.iglob(search_path, recursive=True):
#    print(os.path.basename(i).split('_')[0])
#    break


def five_crops_json(nodes, patch_size, image_size):
    # point : x, y
    x, y = image_size
    reader = NucleiReader()
    res = []
    top_left = reader.crop(nodes, (0, 0), (patch_size, patch_size))
    top_right = reader.crop(nodes, (x - patch_size, 0), (x, patch_size))
    bot_left = reader.crop(nodes, (0, y - patch_size), (patch_size, y))
    bot_right = reader.crop(nodes, (x - patch_size, y - patch_size), (x, y))

    #print(((x - patch_size) / 2, (y - patch_size) / 2),
                        #((x + patch_size) / 2,  (y + patch_size) / 2))
    center = reader.crop(nodes, ((x - patch_size) / 2, (y - patch_size) / 2),
                        ((x + patch_size) / 2,  (y + patch_size) / 2))

    return top_left, top_right, bot_left, bot_right, center

def five_crops(image, patch_size):
    top_left = image[:patch_size, :patch_size]
    top_right = image[:patch_size, -patch_size:]
    bot_left = image[-patch_size:, :patch_size]
    bot_right = image[-patch_size:, -patch_size:]
    #print('bot_Right', bot_right)
    h = image.shape[0]
    w = image.shape[1]
    start_h = int(h / 2 - patch_size / 2)
    start_w = int(w / 2- patch_size / 2)
    center = image[start_h : start_h + patch_size, start_w : start_w + patch_size]
    return top_left, top_right, bot_left, bot_right, center

def crop_image(ori_path, save_path, patch_size=1550):
    search_path = os.path.join(ori_path, '**', '*.png')
    for fp in glob.iglob(search_path, recursive=True):
        #print(fp)
        image = cv2.imread(fp, -1)
        crops = five_crops(image, patch_size)
        for idx, patch in enumerate(crops):
            basename = os.path.basename(fp)
            basename = basename.replace('.png', '_crop_{}.png'.format(idx))
            path = os.path.join(save_path, basename)
            print(path)
            cv2.imwrite(path, patch)
            #print(path)

def crop_json(ori_path, save_path, patch_size=1550):
    search_path = os.path.join(ori_path, '**', '*.json')
    reader = NucleiReader()
    for fp in glob.iglob(search_path, recursive=True):
        #image = cv2.imread(fp, -1)
        nodes = reader.read_json(fp)
        nodes = reader.json2node(nodes, scale=1)
        crops = five_crops_json(nodes, patch_size, (3100, 3100))
        for idx, patch in enumerate(crops):
            basename = os.path.basename(fp)
            basename = basename.replace('.json', '_crop_{}.json'.format(idx))
            path = os.path.join(save_path, basename)
            #print(path)
            nodes = reader.node2json(patch, scale=1)
            print(len(patch), path)
            #reader.save_path(path, nodes)
            reader.save_json(path, nodes)

            #cv2.imwrite(path, patch)

def draw_crop(image_path, json_path, start_point, end_point):
    reader = NucleiReader()
    nodes = reader.read_json(json_path)
    nodes = reader.json2node(nodes, scale=1)
    nodes = reader.crop(nodes, start_point, end_point)
    #nodes = reader.node2json(nodes, scale=1)

    image = cv2.imread(image_path)

    for node in nodes:
        image = cv2.circle(image, tuple(node.centroid), 3, (222, 222, 222), 3)
        bbox = [int(b) for b in node.bbox]
        #print(tuple(bbox[:2]), tuple(bbox[2:]))
        image = cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color=(23, 23, 23), thickness=2)

        #cv2.rectangle(img=image,
        #              pt1=tuple(node.bbox[:2]),
        #              pt2=tuple(node.bbox[2:]),
        #              color=(33, 33, 33),
        #              thickness=3)


    image = cv2.rectangle(image, start_point, end_point, (0, 0, 0), 5)

    return image

#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/images/'
#save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/5Crops'

def read_csv(csv_fp):
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

        return image_names, label

#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/Gleason_masks_train'
#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/Gleason_masks_test_pathologist1'
#save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/5Crops'
#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/json_labels/json/'
#save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/5Crops/'

#imagepath = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/images/ZT199_1_B/ZT199_1_B_8_9.jpg'
#jsonpath = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/json_labels/json/ZT199_1_B_8_9.json'

path_mask = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/5Crops/'
save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/5Crops/'

#path_mask = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/5Crops/'
#save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/5Crops/'

def validate_labels(image_path, label_path):
    import random
    labels = []
    print(image_path)
    for i in glob.iglob(os.path.join(label_path, '**', '*.png'), recursive=True):
        labels.append(i)

    label_fp = random.choice(labels)
    image_fp = os.path.basename(label_fp).replace('.png', '.jpg').replace('mask1_', '').replace('mask_', '')
    image_fp = os.path.join(image_path, image_fp)
    print(label_fp)
    print(image_fp)

    label = cv2.imread(label_fp)
    image = cv2.imread(image_fp)
    mask = np.all(label!=(0,0,0), axis=2)

    mask = dilate(image, mask)
    print(label.shape, image.shape)
    image[mask != 0] = 0
    res = np.hstack([image, label])
    res = cv2.resize(res, (0, 0), fx=0.3, fy=0.3)
    print(res.shape)
    cv2.imwrite('heihei.jpg', res)


#validate_labels(path_mask, save_path)




write_gleason_score(path_mask, save_path)
#crop_image(path, save_path)
#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/Gleason_masks_test_pathologist1'
#crop_image(path, save_path)
#crop_json(path, save_path)
#image = draw_crop(imagepath, jsonpath, (775, 775), (2325, 2325))
#image = draw_crop(imagepath, jsonpath, (0, 0), (3100, 3100))
#image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
#cv2.imwrite('1112.jpg', image)


#img_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/images/ZT199_1_B/ZT199_1_B_7_7.jpg'

#path = os.path.join(path, 'ZT199_1_B_7_7.json')

#start_point = (0, 0)
#end_point = (1000, 1500)


    #a = cv2.resize(image, (0, 0), fy=0.2, fx=0.2)
    #cv2.imwrite('aa.jpg', a)


path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/json_withtypes/json/'
save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/5Crops_json_withtypes/'
#crop_json(path, save_path)

#path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/images/ZT111_4_B_1_14/ZT111_4_B_1_14.jpg'
#image = cv2.imread(path)
##
##print(image.shape)
#images = five_crops(image, 1550)
#count = 0
#for img in images:
#    count += 1
#    print(img.shape)
#    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
#    cv2.imwrite('tmp/crop{}.jpg'.format(count), img)
##
#image = cv2.resize(image, (0,0), fx=0.2, fy=0.2)
#cv2.imwrite('tmp/ori.jpg',image)