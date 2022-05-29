from pathlib import Path
import glob
import os

from skimage.measure import regionprops
import numpy as np
import cv2
from dataflow.stich import JsonFolder, JsonDataset, ImageFolder, ImageDataset


class RawDataReaderCRC:

    def __init__(self, image_folder, label_folder, dataset, return_mask=False):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_names = []
        self.return_mask = return_mask
        search_path = os.path.join(image_folder, '**', '*.png')
        for image_name in glob.iglob(search_path, recursive=True):
            if 'mask' in image_name:
                continue

        #fold_2/2_low_grade/
            #if 'fold_2' not in image_name:
            #    continue
            #if '2_low_grade' not in image_name:
            #    continue
            self.image_names.append(image_name)

        self.dataset = dataset
        print(len(self.image_names))
        #for i in self.image_names:
            #print(i)
        #import sys; sys.exit()

    def __len__(self):
        return len(self.image_names)

    def image2mask_fp(self, image_path):
        rel_image_path = Path(image_path).relative_to(self.image_folder)
        mask_path = os.path.join(self.label_folder, str(rel_image_path).replace('.png', '.npy'))
        return mask_path

    def __getitem__(self, idx):
        image_path = self.image_names[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        bboxes = []
        coords = []

        mask_path = self.image2mask_fp(image_path)
        mask = np.load(mask_path)
        props = regionprops(mask)
        for prop in props:
            bboxes.append([int(c) for c in prop.bbox])
            coords.append([int(c) for c in prop.centroid])

        rel_image_path = str(Path(image_path).relative_to(self.image_folder))

        if not self.return_mask:
            mask = None

        return self.dataset(
                rel_path=rel_image_path,
                image=image,
                bboxes=bboxes,
                coords=coords,
                mask=mask
                )
        #if self.return_mask:
        #    #return image, rel_image_path, bboxes, coords, mask
        #    return rel_image_path, np.concatenate((image, mask), axis=2), bboxes, coords
        #    #return np.rel_image_path, image, bboxes, coords, mask
        #else:
        #    return rel_image_path, image, bboxes, coords

class RawDataReaderCRCRandom:

    def __init__(self, image_folder, label_folder, dataset, return_mask=False):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_names = []
        self.return_mask = return_mask
        search_path = os.path.join(image_folder, '**', '*.png')
        for image_name in glob.iglob(search_path, recursive=True):
            if 'mask' in image_name:
                continue

        #fold_2/2_low_grade/
            #if 'fold_2' not in image_name:
            #    continue
            #if '2_low_grade' not in image_name:
            #    continue
            self.image_names.append(image_name)

        self.dataset = dataset
        print(len(self.image_names))
        #for i in self.image_names:
            #print(i)
        #import sys; sys.exit()

    def __len__(self):
        return len(self.image_names)

    def image2mask_fp(self, image_path):
        rel_image_path = Path(image_path).relative_to(self.image_folder)
        mask_path = os.path.join(self.label_folder, str(rel_image_path).replace('.png', '.npy'))
        return mask_path

    def random_bbox_coordinates(self, image, bboxes, coords):
        h, w = image.shape[:2]
        random_bboxes = []
        random_coords = []

        length = len(bboxes)

        for _ in range(length):
            rand_h = random.choice(range(32, h - 32))
            rand_w = random.choice(range(32, w + 32))
            random_bboxes.append((rand_h, rand_w))




    def __getitem__(self, idx):
        image_path = self.image_names[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        bboxes = []
        coords = []

        mask_path = self.image2mask_fp(image_path)
        mask = np.load(mask_path)
        props = regionprops(mask)
        for prop in props:
            bboxes.append([int(c) for c in prop.bbox])
            coords.append([int(c) for c in prop.centroid])

        rel_image_path = str(Path(image_path).relative_to(self.image_folder))

        if not self.return_mask:
            mask = None
        else:
            raise ValueError(
                'return_mask should be false'
            )

        return self.dataset(
                rel_path=rel_image_path,
                image=image,
                bboxes=bboxes,
                coords=coords,
                mask=mask
                )
        #if self.return_mask:
        #    #return image, rel_image_path, bboxes, coords, mask
        #    return rel_image_path, np.concatenate((image, mask), axis=2), bboxes, coords
        #    #return np.rel_image_path, image, bboxes, coords, mask
        #else:
        #    return rel_image_path, image, bboxes, coords


class RawDataReaderExCRC:

    def __init__(self, image_folder, label_folder, dataset, return_mask=False):
        self.image_folder = image_folder
        self.label_folder = label_folder
        # self.image_names = []
        self.return_mask = return_mask
        # self.stich_image = ImageDataset(self.image_folder, )
        # self.stich_json = JsonDataset(JsonFolder(self.label_folder))

        # image_folder = ImageFolder(image_folder)
        # self.
        self.image_dataset = ImageDataset(ImageFolder(image_folder), 224 * 8)
        # image_path, image = image_dataset[44]
        self.json_dataset =  JsonDataset(JsonFolder(label_folder), 224 * 8)


        # json_folder = JsonFolder(json_path)
        # json_dataset = JsonDataset(json_folder, 224 * 8)
        # json_fp = os.path.join(json_path, os.path.basename(image_path).replace('.png', '.json'))
        # json_label = json_dataset.get_file_by_path(json_fp)

        # image = cv2.resize(image, (0, 0), fx=2, fy=2)
        # image = draw_nuclei(image, json_label)
        # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        # cv2.imwrite('aa.jpg', image)

        self.dataset = dataset
        self.image_names =  self.image_dataset.file_path_lists
        print(len(self.image_names))

        # self.json_dataset = JsonDataset()
        #for i in self.image_names:
            #print(i)
        #import sys; sys.exit()
        # self.iamge_names = file_list

        self.return_mask = return_mask
        self.dataset = dataset
        self.scale = 1


    def __len__(self):
        return len(self.image_names)

    #def image2mask_fp(self, image_path):
    #    rel_image_path = Path(image_path).relative_to(self.image_folder)
    #    mask_path = os.path.join(self.label_folder, str(rel_image_path).replace('.png', '.npy'))
    #    return mask_path

    def format_labels(self, labels, image_size):
        coords = []
        bboxes = []
        if self.return_mask:
            image_size = [s * self.scale for s in image_size]
            mask = np.zeros(image_size, dtype='uint8')

        type_probs = []
        for node in labels:
            cen = node['centroid']
            cen = [int(c // 2) for c in cen]
            cen = cen[::-1]
            coords.append(cen)
            #image = cv2.circle(image, tuple(cen), 3, (0, 200, 0), cv2.FILLED, 1)
            # node['bbox'] : [[rmin, cmin], [rmax, cmax]]
            bbox = node['bbox']
            #print(node['type_prob'])
            # bbox : [min_y, min_x, max_y, max_x]
            #bbox = [b // 2 for b in sum(bbox, [])]
            bbox = [b for b in sum(bbox, [])]
            bbox = [int(b // 2) for b in bbox]
            bboxes.append(bbox)
            # print(bbox, cen)

            #contour = node['contour']
            #node['contour'] = [[c1 // 2, c2 // 2] for [c1, c2] in contour]
            #type_probs.append(node['type_prob'])

            cnt = [[c1 // 2, c2 // 2] for [c1, c2] in node['contour']]

            if self.return_mask:
                cv2.drawContours(mask, [np.array(cnt)], -1, 255, -1)

        # print(coords)
        # print(bboxes)
        if self.return_mask:
            return bboxes, coords, mask
        else:
            return bboxes, coords, None

    def __getitem__(self, idx):
        # image_path = self.image_names[idx]
        # image = cv2.imread(image_path)
        # image = cv2.resize(image, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        # self.img_dataset

        # bboxes = []
        # coords = []

        image_path, image = self.image_dataset[idx]

        json_fp = os.path.join(self.label_folder, os.path.basename(image_path).replace('.png', '.json'))
        json_label = self.json_dataset.get_file_by_path(json_fp)
        bboxes, coords, mask = self.format_labels(json_label, image.shape[:2])
        # print('len(bboxes)', len(bboxes), self.__getitem__)

        # image_path, image = image_dataset[44]
        # self.json_dataset =  JsonDataset(JsonFolder(label_folder), 224 * 8)

        # mask_path = self.image2mask_fp(image_path)
        # mask_path = self.image2mask
        # mask = np.load(mask_path)
        # props = regionprops(mask)
        #for prop in props:
        #    bboxes.append([int(c) for c in prop.bbox])
        #    coords.append([int(c) for c in prop.centroid])

        rel_image_path = str(Path(image_path).relative_to(self.image_folder))

        if not self.return_mask:
            mask = None



        return self.dataset(
                rel_path=rel_image_path,
                image=image,
                bboxes=bboxes,
                coords=coords,
                mask=mask
                )



# if __name__ == '__main__':
#     image_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Images'
#     json_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/Json/EXtended_CRC_Mask'

#     RawDataReaderExCRC(image_path, json_path)