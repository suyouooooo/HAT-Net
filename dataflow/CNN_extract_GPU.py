# encoding: utf-8
import numpy as np
import os
import sys
sys.path.append('../')
from skimage.measure import regionprops
from skimage.filters.rank import entropy as Entropy
from skimage.morphology import disk,remove_small_objects
from common.utils import mkdirs
from common.nuc_feature import nuc_stats_new,nuc_glcm_stats_new
import multiprocessing
import time
import cv2
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable


# Load the pretrained model
model = models.resnet18(pretrained=True)
# model = nn.DataParallel(model).cuda()
model = model.cuda()
# Use the model object to select the desired layer
layer = model._modules.get('avgpool')
# # layer = model._modules.get('avgpool')
# # model._modules.get("0.conv")
# # models._modules["0"]._modules.get("conv")

# Set model to evaluation mode
model.eval()
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.5],
                                 std=[0.23])

to_tensor = transforms.ToTensor()


H, W =3584,3584

def euc_dist(name):
    arr = np.load(name)
    arr_x = (arr[:,0,np.newaxis].T - arr[:,0,np.newaxis])**2
    arr_y = (arr[:,1,np.newaxis].T - arr[:,1,np.newaxis])**2
    arr = np.sqrt(arr_x + arr_y)

    np.save(name.replace('coordinate', 'distance'), arr.astype(np.int16))
    return 0

class DataSetting:
    def __init__(self, label = 'fold_3/3_high_grade'):
        self.dataset = 'CRC'
        self.label  = label
        self.root = '/data/smb/syh/PycharmProjects/CGC-Net/data_su/raw'#'/research/dept6/ynzhou/gcnn/data/raw'
        self.test_data_path = os.path.join(self.root, self.dataset, self.label)
        self.save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_su/proto'#'/research/dept6/ynzhou/gcnn/data/proto'
        self.do_eval = False
        self.test_image_list = os.listdir(self.test_data_path)
        self.test_image_list = [f for f in self.test_image_list if 'png' in f]
        self.test_data = self.test_image_list


class GraphSetting(DataSetting):
    def __init__(self):
        super(GraphSetting, self).__init__()
        self.numpy_list = os.listdir(os.path.join(self.save_path, 'mask',self.dataset, self.label))
        self.numpy_list = [ f for f in self.numpy_list if 'npy' in f]
        self.feature_save_path = os.path.join(self.save_path,'feature', self.dataset, self.label, )
        self.distance_save_path = os.path.join(self.save_path,'coordinate', self.dataset, self.label)
        mkdirs(self.distance_save_path)
        mkdirs(self.feature_save_path)
        mkdirs(self.distance_save_path.replace('coordinate', 'distance'))


def get_vector(arr_image):
    # 1. Load the image with Pillow library
    img = Image.fromarray(np.uint8(arr_image))

    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0).cuda())

    # 3. Create a vector of zeros that will hold our feature vector
    my_embedding = torch.zeros(512)

    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))

    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)

    # 6. Run the model on our transformed image
    model(t_img)

    # 7. Detach our copy function from the layer
    h.remove()

    # 8. Return the feature vector
    return my_embedding.numpy()

def listdir(path):
    list_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        filename = file_path.split('/')[-1]
        list_name.append(filename)
    print(len(list_name))
    return list_name


def _get_batch_features_new(numpy_queue):
    self = GraphSetting()
    done_list = listdir(self.feature_save_path)
    while True:
        name = numpy_queue.get()
        if name != 'end':
            # 判断当前的image是否在list中
            if str(name) in done_list:
                done_list.remove(str(name))
                # info_queue.put(name)
                continue
            # prepare original image
            mask_npy = os.path.join(self.save_path, 'mask', self.dataset, self.label, name)
            print("name: " + str(name))
            # 把分割后的mask npy加载到mask变量中
            mask = np.load(mask_npy)
            # 把过于小的目标去掉
            mask = remove_small_objects(mask, min_size=10, connectivity=1, in_place=True)
            # 把原始图像读取到ori_image中, ori_image.shape(1792,1792,3)
            ori_image = cv2.imread(os.path.join(self.test_data_path, name.replace('.npy', '.png')))
            int_image = cv2.resize(ori_image, (W,H), interpolation=cv2.INTER_LINEAR)
            # 计算图像的信息熵 https://towardsdatascience.com/image-processing-with-python-working-with-entropy-b05e9c84fc36
            # regionprops用来测量标记图像区域的属性，比如连通域的面积，外接矩形的面积，连通域的质心等等
            props = regionprops(mask)
            # 将mask变成一个二值图像，即有细胞核的地方数字为1，其余地方为0
            binary_mask = mask.copy()
            binary_mask[binary_mask>0]= 1
            node_feature = []
            node_coordinate = []

            for prop in props:
                bbox = prop.bbox
                # 提取质心特征，centroid表示质心坐标
                coor = prop.centroid

                single_int = int_image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]

                feature = get_vector(single_int)
                node_feature.append(feature)
                node_coordinate.append(coor)

            node_feature = np.vstack(node_feature)
            # print(node_feature.shape) # (5521, 512)
            node_feature_tensor = torch.from_numpy(node_feature)
            node_feature_tensor = torch.reshape(node_feature_tensor, (1, node_feature_tensor.shape[0], node_feature_tensor.shape[1]))
            b = torch.nn.functional.adaptive_avg_pool2d(node_feature_tensor, (node_feature_tensor.shape[1], 16))
            node_feature = torch.reshape(b, (node_feature_tensor.shape[1], 16))
            node_feature = node_feature.numpy()
            # print(node_feature.shape) # (5521, 16)
            # print(type(node_feature))<class 'numpy.ndarray'>
            node_coordinate = np.vstack(node_coordinate)
            np.save(os.path.join(self.feature_save_path, name), node_feature.astype(np.float32))
            np.save(os.path.join(self.distance_save_path, name), node_coordinate.astype(np.float32))
            euc_dist(os.path.join(self.distance_save_path, name))

            # info_queue.put(name)

        else:
            break

if __name__ == '__main__':
    setting = GraphSetting()
    import queue
    # nameQueue = multiprocessing.Queue()
    # infoQueue = multiprocessing.Queue()

    nameQueue = queue.Queue()
    # infoQueue = queue.Queue()

    for i in setting.numpy_list:
        nameQueue.put(i)
    nameQueue.put('end')
    # Process_C = []
    # for i in range(1):
        # Process_C.append(multiprocessing.Process(target=_get_batch_features_new, args=(nameQueue, infoQueue)))
    _get_batch_features_new(nameQueue)
    # for i in range(1):
    #     Process_C[i].start()

    # total = len(setting.numpy_list)
    # finished = 0
    # while True:
    #     token_info = nameQueue.empty()
    #     finish_name = infoQueue.get(1)
    #     if finish_name in setting.numpy_list:
    #         finished += 1
    #     if finished % 5 == 0:
    #         print('Finish %d/%d'%(finished, total))
    #     if token_info:
    #         print('finished')
    #         time.sleep(10)
    #         for i in range(1):
    #             Process_C[i].terminate()
    #         for i in range(1):
    #             Process_C[i].join()
    #         break