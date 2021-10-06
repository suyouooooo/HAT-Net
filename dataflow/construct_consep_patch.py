import os
import glob
from torch.utils.data import Dataset
from scipy import io
import cv2
import numpy as np

from skimage.measure import label, regionprops




def get_cls_id(cls_patch):
    #print(cls_patch.shape)
    unique_elements, frequency = np.unique(cls_patch, return_counts=True)
    #print(11111111111, frequency, unique_elements)
    sorted_indexes = np.argsort(frequency)[::-1]
    #print(sorted_indexes)

    for idx in sorted_indexes:
        #print('111', unique_elements[idx], frequency[idx])
        #print(unique_elements[idx] == 0, unique_elements[idx])
        if unique_elements[idx] == 0.0:
            continue
        return int(unique_elements[idx])

    print(unique_elements, frequency, cls_patch.shape)
    #import sys
    #sys.exit()
    #for i in sorted_indexes:
    #    i

#def pad(min_x, min_y, max_x, max_y, output_size):
#    if max_x - min_x
colors = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 0),
    (255, 255, 255),
]

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

def get_mask(type_map, cls_id):


        if cls_id == 3 or cls_id == 4:
            mask = (type_map == 3) | (type_map == 4)
        elif cls_id == 0:
            mask = type_map == 0
        elif cls_id == 5 or cls_id == 6 or cls_id == 7:
            mask = (type_map == 5) | (type_map == 6) | (type_map == 7)
        elif cls_id == 2:
            mask = type_map == 2
        else:
            mask = type_map == 1

        return mask

def extract_nuclei(root_path, save_path, patch_size, img_type, image_set):

    #classes = {
    #        1 : 0,
    #        2 : 1,
    #        3 : 2,
    #        4 : 2,
    #        5 : 3,
    #        6 : 3,
    #        7 : 3
    #}
    classes = {
            1 : 0,
            2 : 1,
            3 : 2,
            4 : 2,
            5 : 3,
            6 : 3,
            7 : 3
    for idx, i in enumerate(glob.iglob(os.path.join(CoNSep_PATH, image_set, 'Labels', '*.mat'))):
        label = io.loadmat(i)
        inst_centroid = label['inst_centroid']
        print(i)
        img_fp = i.replace('Labels', img_type).replace('mat', 'png')
        img = cv2.imread(img_fp, -1)
        type_map = label['type_map']
        inst_map = label['inst_map']
        inst_type = label['inst_type']

        unique_elements = np.unique(inst_map)

        for i in unique_elements:
            if i == 0:
                continue
            #print(i)
            #inst = inst_map[inst_map == i]
            x, y = (inst_map == i).nonzero()
            min_x, min_y, max_x, max_y = min(x), min(y), max(x), max(y)
            #print(min_x, min_y, max_x, max_y)
            #print(max_y - min_y,  max_x - min_x)
            #printinst_type[i]

            #mask = inst_map == i
        #print(inst_map.astype('uint8').dtype)
            #props = regionprops(mask.astype('uint8'))
            #print(region.bbox)
            #for prop in props:
            #print(prop.bbox)
                #minr, minc, maxr, maxc = prop.bbox
            #print(len(inst_type), inst_type)
            #print(3333, unique_elements, len(unique_elements))
            #import sys
            #sys.exit()
            cls_id = inst_type[int(i - 1)]
            #mask = (type_map == cls_id) | (type_map == 0)
            mask = get_mask(type_map, cls_id) | get_mask(type_map, 0)
            #mask = (type_map == 7) | (type_map == 0)
            # 1 yellow
            # 2 pink
            # 3 green
            # 4 red
            # 5 blue
            # 6 baby blue
            # 7 yellowish-brown

            #cv2.imwrite('test{}.jpg'.format(idx), img)
            #if idx != 3:
            #continue
            #import sys
            #sys.exit()
            # mask = type_map == 0
            #print('before', max_x - min_x,  max_y - min_y)
            min_x, min_y, max_x, max_y = pad_patch(min_x, min_y, max_x, max_y, patch_size)
            #print('after', max_x - min_x,  max_y - min_y)
            #print(cls_id, colors[int(cls_id[0]) - 1])
            # img =
            #img = cv2.rectangle(img, (min_y, min_x), (max_y, max_x), colors[int(cls_id[0]) - 1], 2)
            #print(img.shape)
            #print(min_x, max_x, min_y, max_y)

            img_patch = img[min_x:max_x, min_y:max_y, :]
            mask_patch = mask[min_x:max_x, min_y:max_y]
            #print(img_mask.shape)
            # img_patch[img_mask == 0] = 0
            #print(img.shape)

            img_name = os.path.basename(img_fp).split('.')[0]
            #print(img_name)
            #print(img_patch.shape)
            #img_name =
            # 0 black
            #img_back = img_patch.copy()
            img_patch = img_patch.copy()
            #print(mask)
            img_patch[~mask_patch] = 0
            #img_back[mask] = 0

            cls_id = classes[cls_id[0]]
            cv2.imwrite(os.path.join(save_path, '{name}_{idx}_{cls_id}_{img_size}_{img_type}.jpg'.format(
                name=img_name,
                idx=int(i),
                #cls_id=int(cls_id[0]),
                cls_id=int(cls_id),
                img_size=patch_size,
                img_type=img_type
            )), img_patch)
            #cv2.imwrite('dataflow/consep_data/{}_{}_{}_mask.jpg'.format(img_name, int(i), int(cls_id[0])), mask * 255)
            #img[mask] = (255,255,255)


        #for patch_idx, cen in enumerate(inst_centroid):
        #    #print(cen)
        #    y, x = cen
        #    y = int(y)
        #    x = int(x)
        #    #print(type_map[y, x])

        #    #patch = img[]
        #    #cen =
        #    img = cv2.rectangle(img, (y - 16, x - 16), (y + 16, x + 16), (0, 0, 255), 2)

        #    #print(type_map.shape)
        #    patch = type_map[max(x-10, 0):x+10, max(y-10, 0):y+10]
        #    #if patch.shape[0] != patch.shape[1]:
        #    #print(patch.shape)
        #    if patch.shape[0] != patch.shape[1]:
        #        print(x-20, x+20, y-20, y+20, x, y, patch.shape)
        #        #print(x,y, patch.size, '1111')
        #    #example = img[y-30:y+30, x-30:x+30]
        #    example = img[x-30:x+30, y-30:y+30]
        #    #print(example.shape)
        #    #cv2.imwrite('_{}.png'.format(patch_idx), example)
        #    cls_id = get_cls_id(patch)
        #    #img = cv2.putText(img, str(cls_id), (y, x), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        #    patch_name = os.path.basename(img_fp)
        #    patch_name = patch_name.replace('.', '_' + str(patch_idx) + '_' + str(cls_id) + '.')

        #    #file_name =
        #    #print(patch_name)
        #    #print(patch_name)
        #    #cv2.imwrite('dataflow/consep_data/{}'.format(patch_name))



        #image_name = os.path.basename(img_fp)
        #print(image_name)
        #cv2.imwrite('dataflow/consep_data/{}'.format(image_name), img)
        #break


image_type = ['Overlay', 'Images']
image_set = ['train', 'test']
CoNSep_PATH = '/data/smb/syh/colon_dataset/CoNSeP'
#SAVE_PATH = '/data/by/tmp/HGIN/dataflow/consep_data'
SAVE_PATH = '/data/by/tmp/HGIN/dataflow/consep_data/{}/{}'.format(image_set[0], image_type[0])
#extract_nuclei(CoNSep_PATH, SAVE_PATH, 64, 'Overlay')
extract_nuclei(CoNSep_PATH, SAVE_PATH, 64, image_type[0], image_set[0])