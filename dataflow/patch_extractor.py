import math
from skimage.feature import greycomatrix, greycoprops


import numpy as np
import torch
import torch_geometric
from torchvision import transforms

import cv2
from skimage.filters.rank import entropy as Entropy
from skimage.morphology import disk,remove_small_objects
from skimage.measure import regionprops, label
from common.nuc_feature import nuc_stats_new,nuc_glcm_stats_new
from torch_geometric.nn import radius_graph
from skimage.feature import greycomatrix


class ImagePatches:
    def __init__(self, rel_path, image, bboxes, coords, mask, patch_size, mean, std):
        self.rel_path = rel_path
        self.image = image
        self.bboxes = bboxes
        self.coords = coords
        #self.hf_transform = hf_transform
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.patch_size = patch_size
        self.mask = mask

    def pad_patch(self, min_x, min_y, max_x, max_y, output_size):
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

class DeepPatches(ImagePatches):
    #def __init__(self, rel_path, image, bboxes, coords, mask, patch_size, mean, std, hf_transform):
    def __init__(self, rel_path, image, bboxes, coords, mask, patch_size, mean, std):
        #super().__init__(rel_path, image, bboxes, coords, mask, patch_size, mean, std, hf_transform)
        super().__init__(rel_path, image, bboxes, coords, mask, patch_size, mean, std)

      #mean = [0.72369437, 0.44910724, 0.68094617] # prostate bgr
        #std = [0.17274064, 0.20472058, 0.20758244] # prosate bgr


        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomChoice([
                #transforms.RandomResizedCrop(patch_size, scale=(0.9, 1.1)),
                transforms.Resize((patch_size, patch_size))
            ]),
            #transforms.RandomApply(torch.nn.ModuleList([
            #        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
            #]), p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, idx):
        bbox = self.bboxes[idx]
        #print(bbox, idx)
        bbox = self.pad_patch(*bbox, self.patch_size)
        patch = self.image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
        patch = self.transforms(patch)
        hand_feat = -1 #None
        return np.array(self.coords[idx]), patch, hand_feat


# class CGCPatches(ImagePatches):
#     def __init__(self, rel_path, image, bboxes, coords, mask, patch_size, mean, std):
#         super().__init__(rel_path, image, bboxes, coords, mask, patch_size, mean, std)

#       #mean = [0.72369437, 0.44910724, 0.68094617] # prostate bgr
#         #std = [0.17274064, 0.20472058, 0.20758244] # prosate bgr


#         #self.transforms = transforms.Compose([
#         #    transforms.ToPILImage(),
#         #    transforms.RandomChoice([
#         #        #transforms.RandomResizedCrop(patch_size, scale=(0.9, 1.1)),
#         #        transforms.Resize((patch_size, patch_size))
#         #    ]),
#         #    #transforms.RandomApply(torch.nn.ModuleList([
#         #    #        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
#         #    #]), p=0.3),
#         #    transforms.ToTensor(),
#         #    transforms.Normalize(mean, std),
#         #])
#         #print(np.unique(self.mask), 'unique')
#         #cv2.imwrite('jjj_mask.jpg', self.mask)
#         self.mask[self.mask > 0] = 1
#         self.mask = np.squeeze(self.mask)
#         #print(self.mask.shape)
#         #print(self.mask.shape)
#         self.mask = remove_small_objects(self.mask, min_size=10, connectivity=1, in_place=True)
#         self.mask = label(self.mask)
#         self.int_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
#         self.entropy = Entropy(self.int_image, disk(3))
#         #print(np.unique(self.mask))
#         #try:
#             #print(np.unique(self.mask))
#         print(rel_path)
#         self.props = regionprops(self.mask)
#         print(len(self.props))
#         #except:
#         #    raise ValueError(np.unique(self.mask))
#         #print('here', len(self.props), self.props[0].bbox, self.props[0].centroid)
#         #self.image[self.mask == 1] = 255
#         #image = cv2.resize(self.image, (0, 0), fx=0.4, fy=0.4)
#         #cv2.imwrite('jjj_mask1.jpg', self.mask / self.mask.max() * 255)
#         #cv2.imwrite('fff.jpg', self.image)
#         #import sys; sys.exit()
#         #self.props = [10] * 1000
#         self.binary_mask = self.mask.copy()
#         self.binary_mask[self.binary_mask>0]= 1
#         #print(len(self.props))

#     #def __len__(self):
#         #return len(self.bboxes)
#     def __len__(self):
#         return len(self.props)

#     def __getitem__(self, idx):

#         prop = self.props[idx]
#         #for i in dir(prop):
#         #    print(i)
#         #bbox = self.bboxes[idx]
#         #bbox = [
#         #    [bbox[0], bbox[1]],
#         #    [bbox[2], bbox[3]]
#         #]
#         nuc_feats = []
#         bbox = prop.bbox
#         #print(bbox)
#         single_entropy = self.entropy[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
#         single_mask = self.binary_mask[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1].astype(np.uint8)
#         if single_mask.shape[1] == 0:
#             print(bbox)

#         if single_mask.shape[0] == 0:
#             print(bbox)
#         #print(if s)

#         single_int = self.int_image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
#         # 提取质心特征，centroid表示质心坐标
#         coor = prop.centroid
#         #coor = self.coords[idx]
#         # single_mask代指mask图像，single_int代指原始的图像，根据两种图像的不同提取出细胞核的16种特征
#         mean_im_out, diff, var_im, skew_im = nuc_stats_new(single_mask, single_int)
#         #print(single_mask.shape, single_int.shape)
#         glcm_feat = nuc_glcm_stats_new(single_mask, single_int) # just breakline for better code
#         glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_ASM = glcm_feat
#         mean_ent = cv2.mean(single_entropy, mask=single_mask)[0]
#         #info = cv2.findContours(single_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         #cnt = info[0][0]
#         #num_vertices = len(cnt)
#         #area = cv2.contourArea(cnt)
#         #hull = cv2.convexHull(cnt)
#         #hull_area = cv2.contourArea(hull)
#         #print(area, hull_area)
#         #if hull_area == 0:
#         #    hull_area += 1
#         #solidity = float(area)/hull_area
#         solidity = prop.solidity
#         area = prop.area
#         #if num_vertices > 4:
#         #    centre, axes, orientation = cv2.fitEllipse(cnt)
#         #    majoraxis_length = max(axes)
#         #    minoraxis_length = min(axes)
#         #else:
#         #    majoraxis_length = 1
#         #    minoraxis_length = 1
#         #major
#         majoraxis_length = prop.major_axis_length
#         minoraxis_length = prop.minor_axis_length
#         orientation = prop.orientation
#         #print(majoraxis_length, minoraxis_length)
#         #perimeter = cv2.arcLength(cnt, True)
#         perimeter = prop.perimeter
#         #eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)
#         eccentricity = prop.eccentricity

#         nuc_feats.append(mean_im_out)
#         nuc_feats.append(diff)
#         nuc_feats.append(var_im)
#         nuc_feats.append(skew_im)
#         nuc_feats.append(mean_ent)
#         nuc_feats.append(glcm_dissimilarity)
#         nuc_feats.append(glcm_homogeneity)
#         nuc_feats.append(glcm_energy)
#         nuc_feats.append(glcm_ASM)
#         nuc_feats.append(eccentricity)
#         nuc_feats.append(area)
#         #if np.isinf(majoraxis_length):
#         #    raise ValueError('majoraxis_length', num_vertices, majoraxis_length, cnt)
#         nuc_feats.append(majoraxis_length)
#         #if np.isinf(minoraxis_length):
#         #    print(minoraxis_length)
#         #    raise ValueError('minoraxis_length')
#         nuc_feats.append(minoraxis_length)
#         nuc_feats.append(perimeter)
#         nuc_feats.append(solidity)
#         nuc_feats.append(orientation)
#         nuc_feats.append(coor)
#         hand_feat = np.hstack(nuc_feats)
#         if np.isnan(hand_feat).sum():
#             print(hand_feat)
#             raise ValueError('has nan')

#         if np.isinf(hand_feat).sum():
#             print(hand_feat)
#             raise ValueError('has inf')
#         #print(hand_feat.shape)
#         #print(hand_feat.shape, 3333)
#         #node_feature.append(feature) # feature : (16:)
#         #hand_feat = np.vstack(node_feature)
#         #node_coordinate.append(coor) # coor: tuple len 2

#         #bbox = self.bboxes[idx]
#         ##print(bbox, idx)
#         #bbox = self.pad_patch(*bbox, self.patch_size)
#         #patch = self.image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
#         ##patch = self.transforms(patch)
#         ##hand_feat = -1 #None
#         #print(hand_feat.shape)
#         patch = -1
#         #print(hand_feat.shape)
#         return np.array(coor), patch, hand_feat


class CGCPatches(ImagePatches):
    def __init__(self, rel_path, image, bboxes, coords, mask, patch_size, mean, std):
        super().__init__(rel_path, image, bboxes, coords, mask, patch_size, mean, std)

      #mean = [0.72369437, 0.44910724, 0.68094617] # prostate bgr
        #std = [0.17274064, 0.20472058, 0.20758244] # prosate bgr


        #self.transforms = transforms.Compose([
        #    transforms.ToPILImage(),
        #    transforms.RandomChoice([
        #        #transforms.RandomResizedCrop(patch_size, scale=(0.9, 1.1)),
        #        transforms.Resize((patch_size, patch_size))
        #    ]),
        #    #transforms.RandomApply(torch.nn.ModuleList([
        #    #        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
        #    #]), p=0.3),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean, std),
        #])
        #print(np.unique(self.mask), 'unique')
        #cv2.imwrite('jjj_mask.jpg', self.mask)
        self.mask[self.mask > 0] = 1
        self.mask = np.squeeze(self.mask)
        #print(self.mask.shape)
        #print(self.mask.shape)
        self.mask = remove_small_objects(self.mask, min_size=10, connectivity=1, in_place=True)
        self.mask = label(self.mask)
        self.int_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.entropy = Entropy(self.int_image, disk(3))
        #print(np.unique(self.mask))
        #try:
            #print(np.unique(self.mask))
        #print(rel_path)
        self.props = regionprops(self.mask)
        #print(len(self.props))
        self.binary_mask = self.mask.copy()
        self.binary_mask[self.binary_mask>0]= 1
        #print(len(self.props))

    #def __len__(self):
        #return len(self.bboxes)
    def __len__(self):
        return len(self.props)

    def __getitem__(self, idx):

        prop = self.props[idx]
        #for i in dir(prop):
        #    print(i)
        #bbox = self.bboxes[idx]
        #bbox = [
        #    [bbox[0], bbox[1]],
        #    [bbox[2], bbox[3]]
        #]
        nuc_feats = []
        bbox = prop.bbox
        #print(bbox)
        single_entropy = self.entropy[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
        single_mask = self.binary_mask[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1].astype(np.uint8)
        if single_mask.shape[1] == 0:
            print(bbox)

        if single_mask.shape[0] == 0:
            print(bbox)
        #print(if s)

        single_int = self.int_image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
        # 提取质心特征，centroid表示质心坐标
        coor = prop.centroid
        #coor = self.coords[idx]
        # single_mask代指mask图像，single_int代指原始的图像，根据两种图像的不同提取出细胞核的16种特征
        mean_im_out, diff, var_im, skew_im = nuc_stats_new(single_mask, single_int)
        #print(single_mask.shape, single_int.shape)
        glcm_feat = nuc_glcm_stats_new(single_mask, single_int) # just breakline for better code
        glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_ASM = glcm_feat
        mean_ent = cv2.mean(single_entropy, mask=single_mask)[0]
        #info = cv2.findContours(single_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cnt = info[0][0]
        #num_vertices = len(cnt)
        #area = cv2.contourArea(cnt)
        #hull = cv2.convexHull(cnt)
        #hull_area = cv2.contourArea(hull)
        #print(area, hull_area)
        #if hull_area == 0:
        #    hull_area += 1
        #solidity = float(area)/hull_area
        solidity = prop.solidity
        area = prop.area
        #if num_vertices > 4:
        #    centre, axes, orientation = cv2.fitEllipse(cnt)
        #    majoraxis_length = max(axes)
        #    minoraxis_length = min(axes)
        #else:
        #    majoraxis_length = 1
        #    minoraxis_length = 1
        #major
        majoraxis_length = prop.major_axis_length
        minoraxis_length = prop.minor_axis_length
        orientation = prop.orientation
        #print(majoraxis_length, minoraxis_length)
        #perimeter = cv2.arcLength(cnt, True)
        perimeter = prop.perimeter
        #eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)
        eccentricity = prop.eccentricity

        nuc_feats.append(mean_im_out)
        nuc_feats.append(diff)
        nuc_feats.append(var_im)
        nuc_feats.append(skew_im)
        nuc_feats.append(mean_ent)
        nuc_feats.append(glcm_dissimilarity)
        nuc_feats.append(glcm_homogeneity)
        nuc_feats.append(glcm_energy)
        nuc_feats.append(glcm_ASM)
        nuc_feats.append(eccentricity)
        nuc_feats.append(area)
        #if np.isinf(majoraxis_length):
        #    raise ValueError('majoraxis_length', num_vertices, majoraxis_length, cnt)
        nuc_feats.append(majoraxis_length)
        #if np.isinf(minoraxis_length):
        #    print(minoraxis_length)
        #    raise ValueError('minoraxis_length')
        nuc_feats.append(minoraxis_length)
        nuc_feats.append(perimeter)
        nuc_feats.append(solidity)
        nuc_feats.append(orientation)
        nuc_feats.append(coor)
        hand_feat = np.hstack(nuc_feats)
        if np.isnan(hand_feat).sum():
            print(hand_feat)
            raise ValueError('has nan')

        if np.isinf(hand_feat).sum():
            print(hand_feat)
            raise ValueError('has inf')
        #print(hand_feat.shape)
        #print(hand_feat.shape, 3333)
        #node_feature.append(feature) # feature : (16:)
        #hand_feat = np.vstack(node_feature)
        #node_coordinate.append(coor) # coor: tuple len 2

        #bbox = self.bboxes[idx]
        ##print(bbox, idx)
        #bbox = self.pad_patch(*bbox, self.patch_size)
        #patch = self.image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
        ##patch = self.transforms(patch)
        ##hand_feat = -1 #None
        #print(hand_feat.shape)
        patch = -1
        #print(hand_feat.shape)
        return np.array(coor), patch, hand_feat


class VGG16Patches(ImagePatches):
    def __init__(self, rel_path, image, bboxes, coords, mask, patch_size, mean, std):
        super().__init__(rel_path, image, bboxes, coords, mask, patch_size, mean, std)
    #def __init__(self, image, coords, bboxes, patch_size):
        """
            bboxes :bbox [min_y, min_x, max_y, max_x]
            coords: (row, col)
        """

        #from model.cpc import get_transforms
        self.image = image
        self.bboxes = bboxes
        self.patch_size = patch_size
        self.coords = coords
        edge_idx = radius_graph(torch.tensor(self.coords), r=self.patch_size, batch=None, loop=True)
        row, col = edge_idx
        self.degree = torch_geometric.utils.degree(col)


        assert self.image.shape[:2] == self.mask.shape[:2]
        #print(len(self.coords), len(self.degree))
        #print(self.degree.max(), self.degree.min(), self.degree.shape, 1111)

        #self.transforms = transforms.Compose([
        #    transforms.Resize((patch_size, patch_size)),
        #    transforms.ToTensor(),
        #    transforms.Normalize(torch.tensor(self.mean), torch.tensor(self.std))
        #])
        #print(self.mean, self.std)

        #self.mean = torch.tensor((0.42019099703461577, 0.41323568513979647, 0.4010048431259079))
       # self.std = torch.tensor((0.30598050258519743, 0.3089986932156864, 0.3054061869915674))

    def resize(self, image):
        size = self.patch_size
        image = cv2.resize(image, (size, size))
        #print('resize image:', image.shape)
        return image

    def to_tensor(self, img):
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img.float() / 255.0
        return img

    def normalize(self, img):

        assert torch.is_tensor(img) and img.ndimension() == 3, 'not an image tensor'

        #if not self.inplace:
            #img = img.clone()

        #mean = torch.tensor(self.mean, dtype=torch.float32)
        #std = torch.tensor(self.std, dtype=torch.float32)
        img.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])

        return img

    def remove_nan(self, data):
        data[np.isnan(data)] = 0
        data[np.isinf(data)] = 0
        return data

    def extract_func(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = gray_image / 255  * 4
        #gray_image = gray_image / (gray_image.max() + 1e-8)
        #gray_image = gray_image.astype('uint8')
        #glcm = greycomatrix(gray_image * mask,  [4], angles=[0, np.pi/2], levels=5)
        #print(np.unique(mask))
        glcm = greycomatrix(gray_image.astype('uint8'),  [1], angles=[0, np.pi/2], levels=5)
        #print(glcm.shape)
        glcm = glcm.flatten()
        mean = image.mean(axis=(0, 1))
        #print(np.concatenate([glcm, mean], axis=-1).shape)
        if np.isnan(glcm).sum() > 0:
            raise ValueError('show not be zero')
        return np.concatenate([glcm, mean], axis=-1)


    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, idx):
        bbox = self.bboxes[idx]
        #print(bbox, idx)
        #bbox = self.pad_patch(*bbox, self.patch_size)
        bbox = self.pad_patch(*bbox, self.patch_size)

        #patch = self.image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1].copy()
        #image = patch[:, :, :3]
        #mask = patch[:, :, 3:].astype('uint8')
        image = self.image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1].copy()
        #print(image.shape)
        #mask = self.mask[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1].copy()
        #print('mask shape', mask.shape, np.unique(mask))
        #mask[mask != 0] = 1
        hand_features = self.extract_func(image)
        #hand_features = self.extract_func(image, mask)
        #print(hand_features.shape, 11)
        #print(coord.shape, 22)
        #hand_features = torch.cat([self.extract_func(image, mask[:, :, 0])])
        #coord.extend(hand_features)
        #coord.append(self.degree[idx])
        #hand_features_and_coord = np.array(coord)
        #hand_features = np.array(coord)
        #print(hand_features.shape, 1111)
        #print(self.degree[idx])
        #print(len(self.coords), self.degree.shape)
        assert len(self.coords) == len(self.degree)
        hand_features = np.hstack([hand_features, self.degree[idx]])
        #print(hand_features.shape, 1111, self.degree[idx])
        #print(np.isnan(hand_features).sum())
        #print()
        #print(hand_features)
        #import sys; sys.exit()
        #hand_features_and_coord = self.remove_nan(hand_features_and_coord)
        #print(hand_features_and_coord.shape, hand_features_and_coord.sum())

        #bbox = self.pad_patch(*bbox, self.patch_size)
        #print(bbox[2] - bbox[0], bbox[3] - bbox[1], self.patch_size, bbox)
        #print(image.shape, 1111111111)
        patch = self.image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1][:, :, :3].copy()
        patch = patch.astype('uint8')
        patch = self.resize(patch)
        patch = self.to_tensor(patch)
        patch = self.normalize(patch)
        #top = max(int((patch.shape[0] - 64) / 2), 0)
        #bottom = max(0, top - patch.shape[0])
        #right = max(int((patch.shape[1] - 64) / 2), 0)
        #left = max(0, right - patch.shape[1])
        #print(top, bottom, left, right)
        #print('before pad', patch.shape)
        #patch = cv2.copyMakeBorder(patch, top, bottom, left, right, cv2.BORDER_REFLECT)
        #print('after pad', patch.shape)
        #print(image.shape)
        #patch = self.transforms(patch.astype('uint8'))
        #print(patch.shape)

        #print(patch.shape)
        return np.array(self.coords[idx]), patch, hand_features
        #return patch, hand_features_and_coord
        #patch = self.transforms(patch)
        #return patch, np.array(self.coords[idx])

    def pad_patch(self, min_x, min_y, max_x, max_y, output_size):
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


class CPCPatches(ImagePatches):
    def __init__(self, rel_path, image, bboxes, coords, mask, patch_size, mean, std):
        super().__init__(rel_path, image, bboxes, coords, mask, patch_size, mean, std)
        """
            bboxes :bbox [min_y, min_x, max_y, max_x]
            coords: (row, col)
        """

        from model.cpc import get_transforms
        #self.image = image
        #self.bboxes = bboxes
        #self.patch_size = patch_size
        #self.coords = coords
        #from model.cpc import network
        self.transforms = get_transforms()
        #mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr
        #std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        #mean = [0.73646324, 0.56556627, 0.70180897] # Expanuke bgr
        #self.std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        #std = [0.18869222, 0.21968669, 0.17277594] # Expanuke bgr

        #self.mask[self.mask > 0] = 1
        self.mask = np.squeeze(self.mask)
        self.mask = remove_small_objects(self.mask > 1, min_size=10, connectivity=1, in_place=True)
        self.mask = label(self.mask)
        #print(np.unique(self.mask))
        #import  sys; sys.exit()
        #self.int_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #self.entropy = Entropy(self.int_image, disk(3))
        #print(np.unique(self.mask))
        #try:
            #print(np.unique(self.mask))
        #print(rel_path)
        self.props = regionprops(self.mask)
        #print(len(self.props))
        #self.binary_mask = self.mask.copy()
        #self.binary_mask[self.binary_mask>0]= 1


    # def extract_func(self, image, mask):
    #     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     mask[mask > 0] = 1
    #     binary_mask = mask

    #     entropy = Entropy(gray_image, disk(3))
    #     mean_im_out, diff, var_im, skew_im = nuc_stats_new(binary_mask, gray_image)
    #     glcm_feat = nuc_glcm_stats_new(binary_mask, gray_image) # just breakline for better code
    #     glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_ASM = glcm_feat
    #     mean_ent = cv2.mean(entropy, mask=gray_image)[0]
    #     info = cv2.findContours(binary_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     cnt = info[0][0]
    #     num_vertices = len(cnt)
    #     area = cv2.contourArea(cnt)
    #     hull = cv2.convexHull(cnt)
    #     hull_area = cv2.contourArea(hull)
    #     if hull_area == 0:
    #         hull_area += 1
    #     solidity = float(area)/hull_area
    #     if num_vertices > 4:
    #         centre, axes, orientation = cv2.fitEllipse(cnt)
    #         majoraxis_length = max(axes)
    #         minoraxis_length = min(axes)
    #     else:
    #         orientation = 0
    #         majoraxis_length = 1
    #         minoraxis_length = 1
    #     perimeter = cv2.arcLength(cnt, True)
    #     eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)


    #     nuc_feats = []
    #     #nuc_feats.append(mean_im_out)
    #     nuc_feats.append(diff)
    #     nuc_feats.append(var_im)
    #     nuc_feats.append(skew_im)
    #     nuc_feats.append(mean_ent)
    #     nuc_feats.append(glcm_dissimilarity)
    #     nuc_feats.append(glcm_homogeneity)
    #     nuc_feats.append(glcm_energy)
    #     nuc_feats.append(glcm_ASM)
    #     nuc_feats.append(eccentricity)
    #     nuc_feats.append(area)
    #     nuc_feats.append(majoraxis_length)
    #     #nuc_feats.append(minoraxis_length) #
    #     nuc_feats.append(perimeter)
    #     #nuc_feats.append(solidity)
    #     #nuc_feats.append(orientation) #
    #     #nuc_feats = np.array(nuc_feats)

    #     #print(nuc_feats.shape)

    #     return nuc_feats

    def __len__(self):
        return len(self.props)

    def __getitem__(self, idx):
        #bbox = self.bboxes[idx]
        prop = self.props[idx]
        bbox = prop.bbox
        coord = prop.centroid
        #print(bbox, idx)
        #bbox = self.pad_patch(*bbox, self.patch_size)
        patch = self.image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1].copy()
        image = patch[:, :, :3]
        #mask = self.mask[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1].copy()
        #coord = self.props[idx]


        area = prop.area
        roundness = (4 * math.pi * prop.area) / (max(prop.perimeter, 1) * max(prop.perimeter, 1))
        eccentricity = prop.eccentricity
        convexity = prop.solidity
        orientation = prop.orientation
        perimeter = prop.perimeter
        majoraxis_length = prop.major_axis_length
        minoraxis_length = prop.minor_axis_length

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = greycomatrix(gray_image, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
        dissimilarity = greycoprops(glcm, prop='dissimilarity')[0, 0]
        homogeneity = greycoprops(glcm, prop='homogeneity')[0, 0]
        energy = greycoprops(glcm, prop='energy')
        ASM  = greycoprops(glcm, prop='ASM')

        hand_feat = [
            area,
            roundness,
            eccentricity,
            convexity,
            orientation,
            perimeter,
            majoraxis_length,
            minoraxis_length,
            dissimilarity,
            homogeneity,
            energy,
            ASM
        ]
        #hand_feat = torch.cat(torch.tensor(hand_feat)).float()
        hand_feat = torch.tensor(hand_feat).float()








        #hand_features = self.extract_func(image, mask[:, :, 0])
        #hand_features = self.extract_func(image, mask)
        #print(hand_features.shape, 11)
        #print(coord.shape, 22)
        #hand_features = torch.cat([self.extract_func(image, mask[:, :, 0])])
        #coord.extend(hand_features)
        hand_features_and_coord = np.array(coord)
        #print(hand_features_and_coord.shape, hand_features_and_coord.sum())

        bbox = self.pad_patch(*bbox, self.patch_size)
        patch = self.image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1].copy()
        #print(bbox[2] - bbox[0], bbox[3] - bbox[1], self.patch_size, bbox)
        #print(image.shape, 1111111111)
        # patch = self.image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1][:, :, :3]
        # top = max(int((patch.shape[0] - 64) / 2), 0)
        # bottom = max(0, top - patch.shape[0])
        # right = max(int((patch.shape[1] - 64) / 2), 0)
        # left = max(0, right - patch.shape[1])
        # #print(top, bottom, left, right)
        # #print('before pad', patch.shape)
        # patch = cv2.copyMakeBorder(patch, top, bottom, left, right, cv2.BORDER_REFLECT)
        #print('after pad', patch.shape)
        #print(image.shape)
        patch = self.transforms(patch)
        #print(patch.shape)

        #return patch, hand_features_and_coord
        #print(hand_feat.shape, patch.shape)

        return np.array(coord), patch, hand_feat


    def pad_patch(self, min_x, min_y, max_x, max_y, output_size):
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
