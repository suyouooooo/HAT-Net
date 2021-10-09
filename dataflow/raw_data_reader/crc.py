from pathlib import Path
import glob
import os

from skimage.measure import regionprops
import numpy as np
import cv2

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
            if 'fold_2' not in image_name:
                continue
            if '2_low_grade' not in image_name:
                continue
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
