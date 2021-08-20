import numpy as np
import os
import sys
#sys.path.append('../')
sys.path.append(os.getcwd())
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
import queue
from torchvision import transforms
from dataflow.consep import ConSep
from torch.nn.functional import adaptive_avg_pool3d







H, W =3584,3584
# H, W = 1792, 1792

def euc_dist(name):
    print(name.replace('coordinate', 'distance'))
    arr = np.load(name)
    arr_x = (arr[:,0,np.newaxis].T - arr[:,0,np.newaxis])**2
    arr_y = (arr[:,1,np.newaxis].T - arr[:,1,np.newaxis])**2
    arr = np.sqrt(arr_x + arr_y)

    np.save(name.replace('coordinate', 'distance'), arr.astype(np.int16))
    return 0

class DataSetting:
    def __init__(self, label = 'fold_3/1_normal'):
        self.dataset = 'CRC'
        self.label  = label
        self.root = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw'
        self.test_data_path = os.path.join(self.root, self.dataset, self.label)
        #self.save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_su/proto'
        #self.save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_res50/proto'
        self.save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_res50/512dim/proto'
        self.do_eval = False
        self.test_image_list = os.listdir(self.test_data_path)
        self.test_image_list = [f for f in self.test_image_list if 'png' in f]
        self.test_data = self.test_image_list
        #self.mask_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/proto/'
        self.mask_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_su/proto'


class GraphSetting(DataSetting):
    def __init__(self, label):
        super(GraphSetting, self).__init__(label=label)
        self.label = label
        self.numpy_list = os.listdir(os.path.join(self.mask_path, 'mask',self.dataset, self.label))
        #print(os.path.join(self.save_path, 'mask',self.dataset, self.label))
        self.numpy_list = [ f for f in self.numpy_list if 'npy' in f]
        self.feature_save_path = os.path.join(self.save_path,'feature', self.dataset, self.label, )
        self.distance_save_path = os.path.join(self.save_path,'coordinate', self.dataset, self.label)
        mkdirs(self.distance_save_path)
        mkdirs(self.feature_save_path)
        mkdirs(self.distance_save_path.replace('coordinate', 'distance'))

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

def _get_batch_features_new(numpy_queue, info_queue, label):
    self = GraphSetting(label=label)
    while True:
        name = numpy_queue.get()
        if name != 'end':
            # prepare original image
            mask_npy = os.path.join(self.mask_path, 'mask', self.dataset, self.label, name)
            # raw_png = os.path.join(self.test_data_path, name.replace('.npy', '.png'))
            mask = np.load(mask_npy)
            #print(mask.shape)
            H,W = mask.shape[0], mask.shape[1]
            mask = remove_small_objects(mask, min_size=10, connectivity=1, in_place=True)
            print(os.path.join(self.test_data_path, name.replace('.npy', '.png')))
            print(os.path.join(self.save_path, 'mask', self.dataset, self.label, name))
            ori_image = cv2.imread(os.path.join(self.test_data_path, name.replace('.npy', '.png')))
            # cv2.COLOR_BGR2GRAY 会让图片变黄，将三维图片转换为二维
            #int_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
            int_image = cv2.resize(ori_image, (W,H), interpolation=cv2.INTER_LINEAR)
            #entropy = Entropy(int_image, disk(3))
            # regionprops用来测量标记图像区域的属性，比如连通域的面积，外接矩形的面积，连通域的质心等等
            props = regionprops(mask)
            # 将mask变成一个二值图像
            #binary_mask = mask.copy()
            #binary_mask[binary_mask>0]= 1
            #disp_to_img = np.array(Image.fromarray(binary_mask))
            node_feature = []
            node_coordinate = []
            count = 0
            imgs = []
            print(len(props))
            for prop in props:
                count += 1
                nuc_feats = []
                bbox = prop.bbox
                #print(bbox)
                #single_entropy = entropy[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
                #single_mask = binary_mask[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1].astype(np.uint8)
                min_y = bbox[0]
                min_x = bbox[1]
                max_y = bbox[2]
                max_x = bbox[3]

                min_x, min_y, max_x, max_y = pad_patch(min_x, min_y, max_x, max_y, 64)
                #print(min_x, min_y, max_x, max_y,  max_x - min_x,  max_y - min_y)

                #single_int = int_image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
                #print(int_image.shape)
                img = int_image[min_y:max_y, min_x:max_x, :]
                img = valid_transforms(img).cuda()
                #output = net(img)
                #print(output.shape)
                imgs.append(img)
                coor = prop.centroid
                node_coordinate.append(coor)

                if count == 3000:
                    batch = torch.stack(imgs)
                    batch = batch.cuda()
                    #print(batch.shape)
                    with torch.no_grad():
                        output = net(batch)
                        output = output.unsqueeze(0)
                        #print(output.shape)
                        output = adaptive_avg_pool3d(output, (512, 1, 1))
                        output = output.squeeze()
                        print('output', output.shape)
                    node_feature.append(output.cpu().numpy())
                    count = 0
                    imgs = []
                #else:
                #    continue
                #print(img.shape)
                # 提取质心特征，centroid表示质心坐标
                #print(bbox, img.shape, coor)
                # single_mask代指mask图像，single_int代指原始的图像，根据两种图像的不同提取出细胞核的16种特征
                #mean_im_out, diff, var_im, skew_im = nuc_stats_new(single_mask,single_int)
                #glcm_feat = nuc_glcm_stats_new(single_mask, single_int) # just breakline for better code
                #glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_ASM = glcm_feat
                #mean_ent = cv2.mean(single_entropy, mask=single_mask)[0]
                #info = cv2.findContours(single_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #cnt = info[0][0]
                #num_vertices = len(cnt)
                #area = cv2.contourArea(cnt)
                #hull = cv2.convexHull(cnt)
                #hull_area = cv2.contourArea(hull)
                ##print(area, hull_area)
                #if hull_area == 0:
                #    hull_area += 1
                #solidity = float(area)/hull_area
                #if num_vertices > 4:
                #    centre, axes, orientation = cv2.fitEllipse(cnt)
                #    majoraxis_length = max(axes)
                #    minoraxis_length = min(axes)
                #else:
                #    orientation = 0
                #    majoraxis_length = 1
                #    minoraxis_length = 1
                #perimeter = cv2.arcLength(cnt, True)
                #eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)
                #nuc_feats.append(mean_im_out)
                #nuc_feats.append(diff)
                #nuc_feats.append(var_im)
                #nuc_feats.append(skew_im)
                #nuc_feats.append(mean_ent)
                #nuc_feats.append(glcm_dissimilarity)
                #nuc_feats.append(glcm_homogeneity)
                #nuc_feats.append(glcm_energy)
                #nuc_feats.append(glcm_ASM)
                #nuc_feats.append(eccentricity)
                #nuc_feats.append(area)
                #nuc_feats.append(majoraxis_length)
                #nuc_feats.append(minoraxis_length)
                #nuc_feats.append(perimeter)
                #nuc_feats.append(solidity)
                #nuc_feats.append(orientation)
                #feature = np.hstack(nuc_feats)
                #print(feature.shape, feature)  (16,)


                #node_feature.append(feature)
                #node_coordinate.append(coor)

            if imgs:
                print(len(imgs))
                batch = torch.stack(imgs)
                batch = batch.cuda()
                #print(batch.shape)
                with torch.no_grad():
                    output = net(batch)
                    output = output.unsqueeze(0)
                    #print(output.shape)
                    output = adaptive_avg_pool3d(output, (512, 1, 1))
                    output = output.squeeze()
                    print(output.shape)
                node_feature.append(output.cpu().numpy())
                count = 0
                imgs = []

            if len(node_feature) == 0:
                pass
            else:
                node_feature = np.vstack(node_feature)
                node_coordinate = np.vstack(node_coordinate)
                print(node_feature.shape)
                print(node_coordinate.shape)
                print('heihei', self.feature_save_path)
                print('name', self.distance_save_path)

                #print('lllllll')
                #print(self.feature_save_path)
                #print(self.distance_save_path)
                #print(name)
                #print(os.path.join(self.feature_save_path, name))
                #print(os.path.join(self.distance_save_path, name))

                # 把特征表示和质心表示都存储起来
                #np.save(os.path.join(self.feature_save_path, name), node_feature.astype(np.float32))
                #np.save(os.path.join(self.distance_save_path, name), node_coordinate.astype(np.float32))

                ## 计算质心之间的euc_dist
                #euc_dist(os.path.join(self.distance_save_path, name))


                info_queue.put(name)
        else:
            break

def network(network_name, num_classes, pretrained):
    if network_name == 'resnet34':
        from model.resnet import resnet34
        net = resnet34(pretrained=pretrained)
        if num_classes != 1000:
            net.reset_num_classes(num_classes)
    elif network_name == 'resnet50':
        from model.resnet import resnet50
        net = resnet50(pretrained=pretrained)
        if num_classes != 1000:
            net.reset_num_classes(num_classes)
    else:
        raise ValueError('network names not suppored')
    return net

def build_feature(label):
    setting = GraphSetting(label=label)
    nameQueue = queue.Queue()
    infoQueue = queue.Queue()

    #print(setting.numpy_list)
    for i in setting.numpy_list:
        nameQueue.put(i)
    nameQueue.put('end')
    _get_batch_features_new(nameQueue, infoQueue, label)

if __name__ == '__main__':
    #mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr  colon
    #std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)     colon

    mean = [0.73646324, 0.56556627, 0.70180897] # bgr  # multi organ
    std = [0.18869222, 0.21968669, 0.17277594] # bgr

    valid_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    #net = network('resnet50', 4, False)
    net = network('resnet50', 5, False)
    #net.load_state_dict(torch.load('/data/by/tmp/HGIN/checkpoint/resnet50/Monday_31_May_2021_00h_40m_26s/17-best.pth'))
    #net.load_state_dict(torch.load('/data/by/tmp/Tensorflow-practice/checkpoint/resnet50/Thursday_05_August_2021_08h_06m_29s'))
    #
    net.load_state_dict(torch.load('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ResNet50/97-best.pth')) # all organ
    net = net.cuda()
    labels = [
        'fold_1/1_normal',
        'fold_1/2_low_grade',
        'fold_1/3_high_grade',
        'fold_2/1_normal',
        'fold_2/2_low_grade',
        'fold_2/3_high_grade',
        'fold_3/1_normal',
        'fold_3/2_low_grade',
        'fold_3/3_high_grade',
    ]

    for label in labels:
        build_feature(label=label)

    #setting = GraphSetting()
    #nameQueue = multiprocessing.Queue()
    #infoQueue = multiprocessing.Queue()

    ##print(setting.numpy_list)
    #for i in setting.numpy_list:
    #    nameQueue.put(i)
    #nameQueue.put('end')
    #_get_batch_features_new(nameQueue, infoQueue)
    #Process_C = []
    #for i in range(4):
    #    Process_C.append(multiprocessing.Process(target=_get_batch_features_new, args=(nameQueue, infoQueue)))
    #for i in range(4):
    #    Process_C[i].start()

    #total = len(setting.numpy_list)
    #finished = 0
    #while True:
    #    token_info = nameQueue.empty()
    #    finish_name = infoQueue.get(1)
    #    if finish_name in setting.numpy_list:
    #        finished += 1
    #    if finished % 5 == 0:
    #        print('Finish %d/%d'%(finished, total))
    #    if token_info:
    #        print('finished')
    #        time.sleep(10)
    #        for i in range(1):
    #            Process_C[i].terminate()
    #        for i in range(1):
    #            Process_C[i].join()
    #        break
