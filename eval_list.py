import os
import csv
import glob
import re
from zipfile import ZipFile

def write_csv(image_names, results):
    header = ['case', 'class']
    #csv_fp = generate_csv_path(label_folder)
    csv_fp = 'pred.csv'
    #print(csv_fp)
    with open(csv_fp, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        count = 0
        for image_name, class_id in zip(image_names, results):
            #print(image_name)
            image_name = os.path.basename(image_name)
            #print(help(re.search))
            image_id = re.search(r'test([0-9]+)\.tif', image_name).group(1)
            #print(image_id)
            row = [image_id, class_id]
            if class_id != 0:
                print(row)

            #count += 1
            #print(row)
            writer.writerow(row)


path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Images/test'
results = [0] * 100


#####################
#no sorted
results[0] = 3
results[1] = 2
results[2] = 3
results[3] = 0
results[4] = 3
results[5] = 2
results[6] = 3
results[7] = 3
########## test acc
results[0] = 3
results[1] = 3
results[2] = 1
results[3] = 3
results[4] = 2
results[5] = 1
results[6] = 0
results[7] = 3
results[8] = 2
results[9] = 2
results[10] = 3
results[11] = 0
results[12] = 0
results[13] = 0
results[14] = 1
results[15] = 0
results[16] = 2
results[17] = 0
results[18] = 0
results[19] = 0
results[20] = 3
results[21] = 2
results[22] = 2
results[23] = 1
###############
results[24] = 1
results[25] = 2
results[26] = 3
results[27] = 2
results[28] = 1
results[29] = 3
results[30] = 2
results[31] = 1
results[32] = 1
results[33] = 2
results[34] = 2
results[35] = 0
results[36] = 1
results[37] = 0
results[38] = 0
results[39] = 0
results[40] = 1
results[41] = 1
results[42] = 0
results[43] = 0
results[44] = 0
results[45] = 3
results[46] = 3
results[47] = 1
results[48] = 0
results[49] = 3
###############
results[50] = 3
results[51] = 0
results[52] = 1
results[53] = 0
results[54] = 2
results[55] = 1
results[56] = 1
results[57] = 1
results[58] = 3
results[59] = 1
results[60] = 1
results[61] = 2
results[62] = 3
results[63] = 3
results[64] = 3
results[65] = 1
results[66] = 3
results[67] = 2
results[68] = 2
results[69] = 3
results[70] = 0
results[71] = 3
results[72] = 3
results[73] = 2
results[74] = 2
results[75] = 3
results[76] = 2
results[77] = 2
results[78] = 1
results[79] = 2
results[80] = 0
results[81] = 2
results[82] = 0
results[83] = 0
results[84] = 1
results[85] = 1
results[86] = 1
results[87] = 3
results[88] = 2
results[89] = 3
###############
results[90] = 3
results[91] = 0
results[92] = 2
results[93] = 2
results[94] = 1
results[95] = 3
results[96] = 2
results[97] = 1
results[98] = 0
results[99] = 0



search_path = os.path.join(path, '**', '*.tif')
file_names = glob.iglob(search_path, recursive=True)
file_names = sorted(file_names)
write_csv(file_names, results)

with ZipFile('predictions.zip', 'w') as predictions:
    predictions.write('pred.csv')