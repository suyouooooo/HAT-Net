
import os
import tarfile
import shutil
import cv2
import glob
import re

from torch.utils.data import Dataset

class ConSep(Dataset):
    def __init__(self, img_lists, transforms=None):
        super().__init__()
        """
        Camvid dataset:https://course.fast.ai/datasets
        or simply wget https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz
        Args:
            data_path: path to dataset folder
            image_set: train datset or validation dataset, 'train', or 'val'
            transforms: data augmentations
        """
        self.transforms = transforms
        self.images = []
        self.labels = []
        self.classes = {
            '1' : 0,
            '2' : 1,
            '3' : 2,
            '4' : 2,
            '5' : 3,
            '6' : 3,
            '7' : 3
        }
        #for img_fp in glob.iglob(os.path.join(root, '**', '*.jpg'), recursive=True):
        with open(img_lists) as f:
            for img_fp in f.readlines():
                img_fp = img_fp.strip()
                self.images.append(cv2.imread(img_fp, -1))
                self.labels.append(self.get_cls_id(img_fp))

        self.transforms = transforms
        self.num_classes = 4
        self.mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr
        self.std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)

    def get_cls_id(self, image_path):
        image = os.path.basename(image_path)
        pattern = r'[a-zA-Z]+_[0-9]+_[0-9]+_([0-9]+)_[0-9]+_[a-zA-Z]+'
        cls_id = re.match(pattern, image)[1]
        #cls_id = self.classes[cls_id]
        return int(cls_id)


    def __len__(self):
        return len(self.images)

    #def _group_ids(self, label):
    #    """Convert 32 classes camvid dataset to 12 classes by
    #    grouping the similar class together
    #    Args:
    #        label: a 32 clasees gt label
    #    Returns:
    #        label: a 12 classes gt label
    #    """

    #    masks = [np.zeros(label.shape, dtype='bool') for i in range(len(self.class_names))]
    #    for cls_id_32 in range(len(self._codes)):
    #        cls_name_32 = self._codes[cls_id_32]
    #        cls_name_12 = self._label_IDs[cls_name_32]
    #        cls_id_12 = self.class_names.index(cls_name_12)
    #        masks[cls_id_12] += label == cls_id_32


    #    for cls_id_12, mask in enumerate(masks):
    #        label[mask] = cls_id_12

    #    return label

    def __getitem__(self, index):

        image = self.images[index]
        #label_path = image_path.replace('images', 'labels').replace('.', '_P.')

        #image = cv2.imread(image_path, -1)
        #label = cv2.imread(label_path, 0)
        label = self.labels[index]


        if self.transforms:
                image = self.transforms(image)

        return image, label

import numpy as np
def compute_mean_and_std(dataset):
    """Compute dataset mean and std, and normalize it
    Args:
        dataset: instance of torch.nn.Dataset
    Returns:
        return: mean and std of this dataset
    """

    mean_r = 0
    mean_g = 0
    mean_b = 0

    #opencv BGR channel
    for img, _ in dataset:
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(dataset)
    mean_g /= len(dataset)
    mean_r /= len(dataset)

    diff_r = 0
    diff_g = 0
    diff_b = 0

    N = 0

    for img, _ in dataset:

        diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
        diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
        diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

        N += np.prod(img[:, :, 0].shape)

    std_b = np.sqrt(diff_b / N)
    std_g = np.sqrt(diff_g / N)
    std_r = np.sqrt(diff_r / N)

    mean = (mean_b.item() / 255.0, mean_g.item() / 255.0, mean_r.item() / 255.0)
    std = (std_b.item() / 255.0, std_g.item() / 255.0, std_r.item() / 255.0)
    return mean, std

#dataset = ConSep(root='/data/by/tmp/HGIN/dataflow/consep_data/train/Images')
#import sys
#sys.path.append(os.getcwd())
#from common.utils import compute_mean_and_std

#print(compute_mean_and_std(dataset))
#image, label = dataset[33]
#cv2.imwrite('ff.png', image)
#import time
#start = time.time()
#print(len(dataset))
#count = 0
#for i in dataset:
#    img, label = i
#    count += 1
#    #print(img.shape, count)
#    #print(i[0].shape, i[1])
#    #pass
##img, label = dataset[]
#
#finish = time.time()
#
#print((finish - start) / len(dataset))