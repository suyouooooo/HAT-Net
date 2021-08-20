import os
import csv
import glob
from shutil import copyfile


csv_fp = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/Gleason_masks_test_pathologist1.csv'

def read_csv(csv_fp):
        res = {}
        with open(csv_fp) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image_name, label = row['image name'], row['gleason score']
                cls_id = int(label) - 5
                #res[image_name] = cls_id
                res[image_name.replace('.jpg', '.json')] = cls_id

        return res

#source_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/images/'
#dest_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images_Aug/test/'
source_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/json_labels/json/'
dest_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json_Aug/test/'
res = read_csv(csv_fp)
#image_names = set(image_names)
count = 0
for i in glob.iglob(os.path.join(source_path, '**', '*.json'), recursive=True):
    base_name = os.path.basename(i)
    if base_name not in res:
        continue
    label = res[base_name]
    base_name = base_name.replace('.', '_grade_{}.'.format(label))
    #print(base_name)
    print(i)
    #count += 1
    print(os.path.join(dest_path, base_name))
    copyfile(i, os.path.join(dest_path, base_name))
    #if base_name not in set(image_names)


    #i.replace('.', '_grade_{}.'.format())

print(count)