import os
import lmdb
import glob
import pickle
#import sys
#sys.path.append(os.getcwd())
from slice_dataset import slice_image, save_patches
import torch

import cv2
#def lmdb_write(lmdb_path, )
class LMDB:
    def __init__(self, lmdb_path):
        #self.lmdb_path = lmdb_path
        map_size = 10 << 40
        print(lmdb_path)
        self.env = lmdb.open(lmdb_path, map_size=map_size)

    def add_files(self, pathes):
        with self.env.begin(write=True) as txn:
            for fp in pathes:
                with open(fp, 'rb') as f:
                    image_buff = f.read()

                basename = os.path.basename(fp)
                txn.put(basename.encode(), image_buff)

class LMDBPt:
    def __init__(self, path, save_path):
        search_path = os.path.join(path, '**', '*.pt')
        map_size = 10 << 40
        self.env = lmdb.open(save_path, map_size=map_size)
        self.file_names = []
        for i in glob.iglob(search_path, recursive=True):
            self.file_names.append(i)
        self.save_path = save_path


    def write(self):
        with self.env.begin(write=True) as txn:
            for idx, fp in enumerate(self.file_names):
                #with open(fp, 'rb') as f:
                    #image_buff = f.read()

                basename = os.path.basename(fp)
                data = torch.load(fp)
                txn.put(basename.encode(), pickle.dumps(data))
                print(idx, fp)

if __name__ == '__main__':
    pt_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Aug'
    save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Aug_LMDB'
    #writer = LMDBPt(pt_path, save_path)
    #writer.write()
    env = lmdb.open(save_path, map_size=1099511627776, readonly=True, lock=False)

    with env.begin(write=False) as txn:
        image_names = [key for key in txn.cursor().iternext(keys=True, values=False)]

    print(len(image_names))
    import random
    image_name = random.choice(image_names)
    with env.begin(write=False) as txn:
        image_data = txn.get(image_name)
        image_data = pickle.loads(image_data)
        print(image_data)




#def extract_class_name

#from lmdb_reader import dataset
#if __name__ == '__main__':
#    #path = '/data/smb/数据集/结直肠/病理学/Extended_CRC/train/Grade2_Patient_007_029810_066073.png'
#    #src_path = '/data/smb/数据集/结直肠/病理学/Extended_CRC/Original_Images/'
#    #dest_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_su/raw/Extended_CRC'
#    #dest_path = 'test_can_be_del2'
#    cache_path = '/data/by/tmp/HGIN/cache1'
#
#    #csv_file = '/data/by/tmp/HGIN/dataflow/extended_crc_fold_info.csv'
#    #split_info = read_csv(csv_file)
#
#    src_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC/fold_2/'
#    lmdb_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC_LMDB/fold_2'
#    os.makedirs(lmdb_path, exist_ok=True)
#    lmdb_dataset = LMDB(lmdb_path)
#
#    #count = 0
#    #image_names = set(dataset.image_names)
#
#    for i in glob.iglob(os.path.join(src_path, '**', '*.png'), recursive=True):
#        #count += 1
#        print(i)
#        if 'mask' in i:
#            continue
#
#        #basename = os.path.basename(i)
#        #if basename in image_names:
#        #    continue
#        #a = image_slicer.slice(i, 32 * 24, save=False)
#        #c = 0
#        #for a1 in a:
#        #    c += 1
#        #    a1.save(
#        #        'test_can_be_del/{}.png'.format(c)
#        #    )
#        #import sys; sys.exit()
#        #print(a[0], type(a[0]))
#        #a[0].save()
#        #print(i)
#        image = cv2.imread(i, -1)
#        #patches = crop_image(image, 224)
#        patches = slice_image(image, 224)
#        image_name = os.path.basename(i)
#        #print(i)
#        #if 'fold_1' in i:
#        #    fold_id = 1
#        #elif 'fold_2' in i:
#        #    fold_id = 2
#        #elif 'fold_3' in i:
#        #    fold_id = 3
#        #else:
#        #    raise ValueError('wrong folder id')
#
#        #print(fold_id)
#        #fold_id = split_info[image_name]
#        #fold_id
#        #if ''
#        #class_name = extract_class_name(image_name)
#        if '1_normal' in i:
#            #class_name = '1_normal'
#            grade_id = 1
#        elif '2_low_grade' in i :
#            #class_name = '2_low_grade'
#            grade_id = 2
#        elif '3_high_grade' in i:
#            #class_name = '3_high_grade'
#            grade_id = 3
#
#        #print(class_name)
#        #fold_folder = 'fold_{}'.format(int(fold_id) + 1)
#        #patch_save_path = os.path.join(dest_path, fold_folder, class_name)
#        #print(class_name, grade_id)
#        #patch_save_path = os.path.join(cache_path, fold_folder, class_name)
#        #print(cache_path)
#        save_patches(patches, cache_path, image_name, grade_id, printable=True)
#        #print(patch_save_path)
#
#
#        ###
#        # read 224 images
#        #for i in glob.iglob(os.path.join(patch_save_path, '**', '**.png'), recursive=True):
#        image_lists = glob.glob(os.path.join(cache_path, '**', '**.png'), recursive=True)
#
#        lmdb_dataset.add_files(image_lists)
#            #print(i)
#
#        for image_fp in image_lists:
#            os.remove(image_fp)
#        ####################
#        #if count == 2:
#
#
#        #    import sys;sys.exit()
#
#        #import sys;sys.exit()
#        #print(len(patches))
#        #print(patches[45]['img'].shape)
#        #print(patches[45]['h'])
#        #print(patches[45]['w'])
#
#        #import sys;sys.exit()
#    #image = cv2.imread(path, -1)
#    #print(image.shape)
#    #images = crop_image(image, 1792, 224)
#    #print(len(images))
#    #print(images[33]['img'].shape)