import json
import os
from pathlib import Path
from functools import partial
import sys
import csv
import random
import pickle
import glob
import string
import time
import cv2
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.getcwd())
from skimage.filters.rank import entropy as Entropy
from skimage.morphology import disk
from common.nuc_feature import nuc_stats_new,nuc_glcm_stats_new
import sys
from dataflow.graph_transform import avg_pooling
#sys.setrecursionlimit(1000)


#sys.path.append('../')

#import lmdb
import torch
from torch.utils.data import DataLoader, Dataset
#from torch.nn.functional import adaptive_avg_pool3d, adaptive_avg_pool2d
#from torchvision import transforms
#from torch_geometric.data import Data
#from torch_geometric.nn import radius_graph
#from torch_cluster import grid_cluster
#from torch_scatter import scatter
#from torch_geometric.nn.pool.consecutive import consecutive_cluster
#from torch_geometric.nn.pool.pool import pool_pos


import cv2
import numpy as np
from skimage.measure import regionprops


from stich import JsonFolder, JsonDataset, ImageFolder, ImageDataset
from patch_extractor import DeepPatches, CGCPatches, VGG16Patches
from raw_data_reader import RawDataReaderProsate5CropsAug, RawDataReaderBACH, RawDataReaderBACHTestSet, RawDataReaderCRC
from deep_extractor import ExtractorResNet50, ExtractorVGG
from cell_graph_dataset import CellGraphPt


def vis(img_fp, json_fp):
    #print(img_fp)
    image = cv2.imread(img_fp)
    #image = img_fp
    #print(json_fp)
    with open(json_fp, 'r') as f:
        res = json.load(f)
        #print(res.keys())
        #print(res['nuc']['2175'])
        print('nuclei number', len(res['nuc'].keys()))
        for node in res['nuc'].keys():
            cen = res['nuc'][node]['centroid']
            cen = [int(c) for c in cen]
            bbox = res['nuc'][node]['bbox']
            #type_id = res['nuc'][node]['type']
            #image = cv2.circle(image, tuple(cen), 3, (0, 30 * type_id, 10 * type_id), cv2.FILLED, 3)
            image = cv2.circle(image, tuple(cen), 3, (0, 255, 0), cv2.FILLED, 3)

            store_bbox = bbox
            bbox = [b for b in sum(bbox, [])]
            #print('after:')
            if bbox[2] - bbox[0] <= 0:
                print(bbox, store_bbox, '1')
                image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (0, 255, 0), 2)
                continue
            if bbox[3] - bbox[1] <= 0:
                print(bbox, store_bbox, 2)
                image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (0, 255, 0), 2)
                continue

            if min(bbox) < 0:
                print(bbox, store_bbox, 3)
                image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (0, 255, 0), 2)
                continue

            image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (255, 0, 0), 2)

            #image = cv2.rectangle()


    return image

#### generating features


class TorchWriter:
    def __init__(self, save_path):
        self.save_path = save_path
        self.thread_pool = ThreadPoolExecutor(4)

    def write(self, pair):
        name, val = pair
        name = name.split('.')[0] + '.pt'
        save_path = os.path.join(self.save_path, name)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #print(save_path)
        torch.save(val, save_path)

    def add_pair(self, pairs):
        #for pair in pairs:
            #write(pair)

        self.thread_pool.map(self.write, pairs)

#def write_data(save_path, data):
#    #print(save_path)
#    save_path = os.path.join(save_path, os.path.basename(data.path.replace('.jpg', '.pt')))
#    #print(111,  save_path)
#    #print(data)
#    #sys.exit()
#    if len(data.x) <= 5:
#        return
#
#    #print(save_path)
#    torch.save(data.clone(), save_path)

class CellGraphWriter:
    def __init__(self, save_path):
        self.save_path = save_path
        self.thread_pool = ThreadPoolExecutor(4)

    def write_sample(self, data):

        #save_path = os.path.join(self.save_path, os.path.basename(data.path.replace('.jpg', '.pt')))
        #data.path : relative path in dataset folder
        save_path = os.path.join(self.save_path, data.path)
        #print(111,  save_path)
        #print(data)
        #sys.exit()
        if len(data.x) <= 5:
            return

        #print(save_path)
        #print(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #print(save_path)
        torch.save(data.clone(), save_path)
        #print(save_path)
        #import sys; sys.exit()

    def write_batch(self, batch):
        self.thread_pool.map(
            self.write_sample,
            [batch[i] for i in range(batch.num_graphs)]
        )



#def write_batch(save_path, batch):
#    for i in range(batch.num_graphs):
#        write_data(save_path, batch[i])

def generate_features_batch(data_loader, num_feats, rel_pathes, writer, deep_extractor):
    node_features = []
    node_coords = []
    for idx, (coords, images, hand_feats) in enumerate(data_loader):
        images_mask = images == -1
        hand_feats_mask = hand_feats == -1

        # images not empty
        if images_mask.sum() != len(images):
            deep_feats = deep_extractor(images)

            #if hand_feats is not None:
            # deep and hand features
            if hand_feats_mask.sum() != len(hand_feats):
                output = np.hstack([deep_feats , hand_feats])

            ## only deep feature
            # hand_feats is None
            else:
                output = deep_feats


        else:
            # no deep feature, only hand features
            #if hand_feats is not None:
            #print('only hand-crafted features')
            if hand_feats_mask.sum() != len(hand_feats):
                output = hand_feats

        #print(coords.shape)

        #print(output.shape, coords.shape)
        node_features.append(output)
        node_coords.append(coords)

    node_features = np.vstack(node_features)
    node_coords = np.vstack(node_coords)
    print('----------------')
    print(node_features.shape)
    print(node_coords.shape)
    assert len(node_features) == len(node_coords)
    prev = 0
    lmdb_pair = []
    for rel_path, num_feat in zip(rel_pathes, num_feats):
        #print(rel_path, num_feat)

        sub_feat = node_features[prev : prev+num_feat]
        sub_coord = node_coords[prev : prev+num_feat]
        #print(sub_feat.shape, sub_coord.shape)
        #print(sub_feat.shape)
        #print(sub_coord.shape)
        #print(sub_feat.dtype, 'before')
        val = {
            'feat' : torch.tensor(sub_feat).float(),
            'coord' : torch.tensor(sub_coord)
        }
        #print(val['feat'].dtype)
        #val = pickle.dumps(val)
        #rel_path = rel_path.replace('.jpg', '.npy')
        rel_path = rel_path.split('.')[0] + '.npy'
        #print(rel_path, sub_feat.shape, sub_coord.shape)
        prev += num_feat
        lmdb_pair.append([rel_path, val])

    print('writting to disk......')
    writer.add_pair(lmdb_pair)


def feat_pipline(patch_dataset, deep_extractor, save_path, num_epoches, data_writer):
    ####
    # read raw images frodata_loaderm disk
    print(len(patch_dataset))
    #batch_size = 128  # batch size for raw images
    #batch_size = 16  # batch size for raw images
    batch_size = 8  # batch size for raw images
    patch_dataset_loader = torch.utils.data.DataLoader(patch_dataset, batch_size=batch_size, num_workers=2, collate_fn=lambda x : x)
    print(patch_dataset_loader)

    start = time.time()
    for idx, pds in enumerate(patch_dataset_loader):
        # patch_datasets, a list of patch_datsets

        #print(idx * batch_size + batch_size)
        #if idx * batch_size + batch_size != 15409:
            #continue
        num_feats = []
        rel_pathes = []
        for pd in pds:
            num_feats.append(len(pd))
            rel_pathes.append(pd.rel_path)

        # generate patches
        print('generating patches')
        pds = torch.utils.data.ConcatDataset(pds)
        pd_loader = DataLoader(pds, num_workers=4, batch_size=128 * 16 * 2, shuffle=False)  # batch size for nuclei patches
        #pd_loader = DataLoader(pds, num_workers=4, batch_size=1, shuffle=False)  # batch size for nuclei patches
        #print('dataset size:', len(patch_dataset), sum(num_feats))

        for epoch in range(num_epoches):
            save_path_epoch = os.path.join(save_path, str(epoch))
            os.makedirs(save_path_epoch, exist_ok=True)
            print('saving data to {}.....'.format(save_path_epoch))
            writer = data_writer(save_path_epoch)
            # generate_features
            generate_features_batch(pd_loader, num_feats, rel_pathes, writer, deep_extractor)

        finish = time.time()
        print('[{}/{}], speed:{:2f}'.format(idx * batch_size + batch_size, len(patch_dataset), (idx * batch_size + batch_size) / (finish - start)))

#def write_data(save_path, data):
#    #print(save_path)
#    save_path = os.path.join(save_path, os.path.basename(data.path.replace('.jpg', '.pt')))
#    #print(111,  save_path)
#    #print(data)
#    #sys.exit()
#    if len(data.x) <= 5:
#        return
#
#    #print(save_path)
#    torch.save(data.clone(), save_path)


def cell_graph_pipline(data_set, writer):

    from torch_geometric.data import DataLoader
    data_loader = DataLoader(data_set, num_workers=4, batch_size=32)

    start = time.time()
    for idx, b in enumerate(data_loader):
            #print(b)
            #print(33333, save_path)
            #import sys; sys.exit()
            print('writing to disk....')
            writer.write_batch(b)
            finish = time.time()
            avg_speed = (idx + 1) * len(b) / (finish - start)
            print('[{}/{}]....avg speed: {:2f}'.format(idx + 1, len(data_loader), avg_speed * len(b)))

    print('total time consumed: {:2f}'.format(time.time() - start))

def validate_image(image_folder, json_folder):

    ext = 'tif'
    images = glob.glob(os.path.join(image_folder, '**', '*.{}'.format(ext)), recursive=True)

    import random
    idx = random.choice(range(len(images)))
    #idx = 1985
    #idx = 1437
    #idx = 2450
    #idx = 2132
    print(idx)
    image = images[idx]
    json_basename = os.path.basename(image)
    json_basename = json_basename.replace('.{}'.format(ext), '.json')
    json_path = os.path.join(json_folder, json_basename)
    print(image)
    print(json_path)
    image = vis(image, json_path)

    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite('test_448_single_process.jpg', image)


    #dataset = Prosate5CropsAugCPC(image_folder, json_folder, return_mask=False)

    #count = 0
    #for idx, bbox in enumerate(bboxes):
    #    assert bbox[0] < bbox[2]
    #    assert bbox[1] < bbox[3]
    #    if type_probs[idx] < 0.5:
    #        count += 1
    #        image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (0, 9, 100), 2)
    #    else:
    #        image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (0, 255, 0), 2)

    ##for idx, coord in enumerate(coords):
    ##    image = cv2.circle(image, tuple(coord[::-1]), 3, (0, 200, 0), cv2.FILLED, 1)

    #image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
    #print(count)
    #cv2.imwrite('fff.jpg', image)

def main():

    #image_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/5Crops_Aug/'
    #json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/5Crops_json_withtypes_Aug/'

    image_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC'
    json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/proto/mask/CRC/shaban-cia'
    #image_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Images_Aug'
    #json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Json_Aug_withtypes'
    #image_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Images/test'
    #json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Json/withtypes/test/json'
    #json_folder = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Json_Aug'

    #validate_image(image_folder, json_folder)
    #import sys; sys.exit()

    #save_path = '/data/hdd1/by/HGIN/tttt_del'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Feat_Aug/hatnet2048dim'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Feat/test/cgc16dim'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Feat/5Crops_Aug_CGC16dim'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Feat/VGGUet_438dim'
    save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Feat/PanNukeEx6Classes'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Feat_Aug/cgc16dim/'
    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Feat/test/hatnet2048dim'

    #patch_size, mean, std, hf_transform

    # mean and std for deep learning image transforms
    ############################
    #mean = [0.72369437, 0.44910724, 0.68094617] # prostate bgr
    #std = [0.17274064, 0.20472058, 0.20758244] # prosate bgr
    mean = [0.73646324, 0.56556627, 0.70180897] # Expanuke bgr /bach
    std = [0.18869222, 0.21968669, 0.17277594] # Expanuke bgr



    #mean = (0.42019099703461577, 0.41323568513979647, 0.4010048431259079) # vgg16unet monusac bgr
    #std = (0.30598050258519743, 0.3089986932156864, 0.3054061869915674) # vgg16unet monusac bgr


    ############################

    # hand crafted feature transform
    #########################################
    #hcf_transform = None
    ################################

    ####################################################
    # read raw images and labels
    #raw_data_reader = RawDataReaderProsate5CropsAug
    raw_data_reader = RawDataReaderCRC
    #raw_data_reader = RawDataReaderBACH
    #raw_data_reader = partial(RawDataReaderBACHTestSet, csv_file='/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/pred.csv')
    #############################################


    #################################
    # extract patches from raw images
    patch_extractor = DeepPatches  # only deep learning transformations
    #patch_extractor = CGCPatches
    #patch_extractor = VGG16Patches
    #####################################

    ################################
    # feature extraactor deep learning
    #weight_file = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ResNet50/81-best.pth' prostate
    #weight_file = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ResNet50/60-best.pth' # bach
    #weight_file = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/checkpoints/vggunet/183-best.pth' # bach
    weight_file = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ResNet50/9-best.pth' # crc pannuke
    num_classes = 6 # pannuke extended
    #num_classes = 5 # vggunet monsac datset
    #output_dim = None
    output_dim = 16
    #deep_extractor = ExtractorResNet50(weight_file, num_classes, output_dim)
    #deep_extractor = None
    #deep_extractor = ExtractorVGG(weight_file, num_classes, output_dim)
    ###############################


    # writer:
    writer = TorchWriter


    print('---------------------------')
    patch_size = 64
    #patch_size = 71 vggunet
    #patch_dataset = partial(patch_extractor, patch_size=patch_size, mean=mean, std=std)
    #raw_data_reader = raw_data_reader(image_folder, json_folder, patch_dataset, return_mask=False)
    #raw_data_reader = raw_data_reader(image_folder, json_folder, patch_dataset, return_mask=True)

    #feat_pipline(raw_data_reader, deep_extractor, save_path, 1, writer)

    #######################################s######################################################
    #######################################s######################################################
    #######################################s######################################################
    #######################################s######################################################
    # genreate cell graph
    #cell_graph_save_path = '/data/hdd1/by/HGIN/ttt_del1'
    #cell_graph_save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/Aug/hatnet2048dim/'
    #cell_graph_save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/test/cgc16dim'
    #cell_graph_save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Cell_Graph/5Crops_Aug_CGC16dim'
    #cell_graph_save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/VGGUet_438dim'
    cell_graph_save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Cell_Graph/PanNukeEx6classes/proto/fix_avg_cia_knn/'
    #cell_graph_save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/Aug/cgc16dim'
    #cell_graph_save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/test/hatnet2048dim/'
    # transforms
    transforms = [partial(avg_pooling, size=64)]


    #save_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Feat/PanNukeEx6Classes/0'

    dataset = CellGraphPt(save_path, transforms=transforms)
    writer = CellGraphWriter(cell_graph_save_path)
    cell_graph_pipline(dataset, writer)

    ########################

    ########################



if __name__ == '__main__':
    main()


    import sys; sys.exit()