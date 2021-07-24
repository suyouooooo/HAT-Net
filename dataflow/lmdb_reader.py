import os
import pickle
import string
import io

import numpy as np
import lmdb
import cv2
from PIL import Image


class LMDB:
    def __init__(self, path, transform, return_path=False):
        self.env = lmdb.open(path, map_size=1099511627776, readonly=True, lock=False)
        #with self.env.begin(write=False) as txn:
        #    self.image_names= [key.decode() for key in txn.cursor().iternext(keys=True, values=False)]

        cache_file = '_cache_' + ''.join(c for c in path if c in string.ascii_letters)
        cache_path = os.path.join(path, cache_file)
        if os.path.isfile(cache_path):
            self.image_names = pickle.load(open(cache_path, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.image_names = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.image_names, open(cache_path, "wb"))
        self.transform = transform
        self.return_path = return_path


    def __len__(self):
        return len(self.image_names)

    def class_id(self, path):
        NotImplementedError


    #def __getitem__(self, idx):
    #    NotImplementedError
    def __getitem__(self, idx):

        image_name = self.image_names[idx]
        with self.env.begin(write=False) as txn:
            image_data = txn.get(image_name)
            #image = np.frombuffer(image_data, np.uint8)
            #image = cv2.imdecode(image, -1)
            image = Image.open(io.BytesIO(image_data))
            label = self.class_id(image_name.decode())


        if self.transform:
            image = self.transform(image)

        if self.return_path:
            return image, label, image_name.decode()
        else:
            return image, label
        #image_name = self.image_names[idx]

        #with self._env.begin(write=False) as txn:
        #    image_data = txn.get(image_name.encode())
        #    print(image_data)
        #    return



class CRC(LMDB):
    def __init__(self, path, transform=None):
        super().__init__(path=path, transform=transform)

    def class_id(self, path):
        #print(path)
        grade_id = path.split('_grade_')[1][0]
        #print(grade_id)
        return int(grade_id) - 1




#del_files = []
#with open('junk_can_be_del') as f:
#    for line in f.readlines():
#        del_files.append(line.strip())

#print(len(del_files))
#for i in del_files:
#    print(i)


#lmdb_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC_LMDB/fold_3'
#env = lmdb.open(lmdb_path, map_size=999999999999)
#
#    #def add_files(self, pathes):
#with env.begin(write=True) as txn:
#    #for
#    for fp in del_files:
#        buff = txn.get(fp.encode())
#        #print(type(buff))
#        #if buff is None:
#        txn.delete(fp.encode())
#            #print('fffffffffff')

#import sys;sys.exit()
dataset = CRC('/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC_LMDB/fold_3')
print(len(dataset))

#files = os.listdir('/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC/fold_3/1_normal/*.png')
#print(len(files))
#files.extends(os.listdir('/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC/fold_3/2_low_grade/*.png'))
#print(len(files))
#files.extends(os.listdir('/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC/fold_3/3_high_grade/*.png'))
#print(len(files))

#c = dataset.image_names
##if 'H09-00622_A2H_E_1_4_grade_1_2241_1121.png'.encode() in c:
##    print('ffffffffffffffffffffffff')
##import sys; sys.exit()
##print(len(c))
#for i in c:
#    #print(i.decode())
#    if 'H09-00622_A2H_E_1_4_grade_1_2241_1121' in i.decode():
#        print(i.decode())
    #print(i[2])
    #print(i)
    #pass

#import sys; sys.exit()
#from torch.utils.data import DataLoader
#
#dataloder = DataLoader(dataset, num_workers=2, batch_size=10)
#
#for i in dataloder:
#    image, label, path = i
#    print(image, label, path)
#    break
#    pass


#count = 0
#print(len(dataset))
image, label = dataset[533333]

#for i in dataset:
#
#    image, label = i
#    print(label)
#    count += 1
#cv2.imwrite('tmp/test{}.png'.format(1), image)
image.save('tmp/test{}.png'.format(1))
