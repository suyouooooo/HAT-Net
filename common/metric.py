import numpy as np
from collections import defaultdict
import sklearn.metrics as metrics
import pdb
CROSS_VAL = {1:3,2:2,3:1}
GROUND_TRUTH = {
    1:['H09-18586_A2H_E_1_3_grade_1', 'H08-144335_A1H&E_1_4_grade_1', 'H09-00622_A2H_E_1_4_grade_1', 'H08-50593_A1H&E_1_2_grade_1', 'H08-144335_A1H&E_1_2_grade_1', 'H08-68294_A1H&E_1_3_grade_1', 'H09-16145_A2H_E_1_1_grade_1', 'H10-24087_A2H_E_1_1_grade_1', 'H09-23100_A1H_E_1_2_grade_1', 'H09-02883_A2H_E_1_5_grade_1', 'H09-12618_A1H_E_1_2_grade_1', 'H09-00622_A2H_E_1_3_grade_1', 'H09-00320_A2H_E_1_2_grade_1', 'H08-36034_A1H&E_1_2_grade_1', 'H09-11303_A2H&E_1_2_grade_1', 'H09-00320_A2H&E_1_3_grade_1', 'H09-00622_A2H&E_1_6_grade_1', 'H08-57464_A1H&E_2_1_grade_1', 'H09-00804_A2H_E_1_2_grade_1', 'H09-12618_A1H_E_1_1_grade_1', 'H08-19835_A1H&E_1_5_grade_1', 'H09-18586_A2H&E_1_13_grade_1', 'H08-144335_A1H&E_1_3_grade_1', 'H09-18586_A2H&E_1_12_grade_1',
       'H09-00804_A2H_E_1_3_grade_2', 'H06-05168_A5H_E_1_1_grade_2', 'H06-02141_A3H_E_1_1_grade_2', 'H06-04442_A5H_E_1_2_grade_2', 'H06-04116_A6H_E_1_2_grade_2', 'H09-00822_A1H_E_1_2_grade_2', 'H09-00320_A2H_E_1_1_grade_2', 'H09-00107_A1H_E_1_4_grade_2', 'H09-00622_A2H_E_1_1_grade_2', 'H06-05790_A4H_E_1_1_grade_2', 'H06-05658_A5H_E_1_1_grade_2',
       'H09-02883_A2H_E_1_1_grade_3', 'H09-24359_A2H_E_1_1_grade_3', 'H09-02883_A2H_E_1_2_grade_3', 'H09-15863_A2H_E_1_7_grade_3', 'H09-24359_A2H_E_1_5_grade_3', 'H09-18586_A2H_E_1_6_grade_3', 'H09-15863_A2H_E_1_8_grade_3', 'H09-16289_A1H_E_1_1_grade_3', 'H09-18586_A2H_E_1_9_grade_3'],

    2:[
       'H09-18586_A2H_E_1_8_grade_1', 'H09-07547_A1H_E_1_1_grade_1', 'H09-16145_A2H&E_1_3_grade_1', 'H09-15863_A2H_E_1_1_grade_1', 'H09-00107_A1H&E_1_7_grade_1', 'H09-00107_A1H_E_1_2_grade_1', 'H09-00622_A2H&E_1_5_grade_1', 'H08-68294_A1H&E_1_5_grade_1', 'H08-68294_A1H&E_1_11_grade_1', 'H09-03067_A2H_E_1_1_grade_1', 'H09-16145_A2H_E_1_2_grade_1', 'H08-81054_A1H&E_1_2_grade_1', 'H08-68294_A1H&E_1_9_grade_1', 'H08-19835_A1H&E_1_3_grade_1', 'H09-12618_A1H&E_1_3_grade_1', 'H08-50593_A1H&E_1_3_grade_1', 'H08-81054_A1H&E_1_3_grade_1', 'H08-19835_A1H&E_1_4_grade_1', 'H09-23100_A1H_E_1_1_grade_1', 'H08-68294_A1H&E_1_4_grade_1', 'H08-68294_A1H&E_1_2_grade_1', 'H09-23100_A1H&E_1_5_grade_1', 'H09-03067_A2H&E_1_2_grade_1',
       'H06-04767_B5H_E_1_1_grade_2', 'H06-03838_A6H_E_1_1_grade_2', 'H06-04767_B5H_E_1_2_grade_2', 'H06-04255_A5H_E_1_1_grade_2', 'H06-04255_A5H_E_1_3_grade_2', 'H06-04116_A6H_E_1_1_grade_2', 'H09-01265_A2H_E_1_1_grade_2', 'H06-04442_A5H_E_1_1_grade_2', 'H09-00822_A1H_E_1_3_grade_2', 'H09-00622_A2H_E_1_2_grade_2', 'H06-04442_A5H_E_1_3_grade_2',
'H09-24359_A2H_E_1_6_grade_3', 'H09-02883_A2H_E_1_6_grade_3', 'H09-15863_A2H_E_1_2_grade_3', 'H09-18586_A2H_E_1_7_grade_3', 'H09-02883_A2H_E_1_4_grade_3', 'H09-18586_A2H_E_1_5_grade_3', 'H09-15863_A2H_E_1_3_grade_3', 'H09-18586_A2H_E_1_4_grade_3', 'H09-15863_A2H_E_1_6_grade_3', 'H09-00804_A2H_E_1_1_grade_3', 'H09-23100_A1H_E_1_4_grade_3', 'H09-16289_A1H_E_1_2_grade_3'
],
  3:['H08-50593_A1H&E_1_4_grade_1', 'H08-68294_A1H&E_1_10_grade_1', 'H08-50593_A1H&E_1_1_grade_1', 'H09-07547_A1H&E_1_2_grade_1', 'H08-50593_A1H&E_1_5_grade_1', 'H09-00107_A1H&E_1_5_grade_1', 'H09-00107_A1H&E_1_6_grade_1', 'H10-24087_A2H_E_1_2_grade_1', 'H09-11303_A2H_E_1_1_grade_1', 'H08-19835_A1H&E_1_1_grade_1', 'H09-11303_A2H&E_1_3_grade_1', 'H08-81054_A1H&E_1_1_grade_1', 'H09-18586_A2H&E_1_11_grade_1', 'H08-160413_A1H&E_1_1_grade_1', 'H09-00107_A1H_E_1_1_grade_1', 'H08-144335_A1H&E_1_1_grade_1', 'H08-36034_A1H&E_1_1_grade_1', 'H09-00622_A2H&E_1_7_grade_1', 'H08-57464_A1H&E_2_2_grade_1', 'H08-19835_A1H&E_1_2_grade_1', 'H08-68294_A1H&E_1_6_grade_1', 'H08-68294_A1H&E_1_7_grade_1',
     'H09-02464_A2H_E_1_1_grade_2', 'H06-05869_A5H_E_1_2_grade_2', 'H06-04255_A5H_E_1_2_grade_2', 'H09-01648_A2H_E_1_1_grade_2', 'H06-02141_A3H_E_1_3_grade_2', 'H06-05658_A5H_E_1_2_grade_2', 'H06-05869_A5H_E_1_1_grade_2', 'H09-01648_A2H_E_1_2_grade_2', 'H06-02141_A3H_E_1_2_grade_2', 'H09-00107_A1H_E_1_3_grade_2', 'H09-00822_A1H_E_1_1_grade_2',
'H09-24359_A2H_E_1_3_grade_3', 'H09-02883_A2H_E_1_3_grade_3', 'H09-24359_A2H_E_1_2_grade_3', 'H09-15863_A2H_E_1_9_grade_3', 'H09-15863_A2H_E_1_5_grade_3', 'H09-18586_A2H_E_1_1_grade_3', 'H06-05790_A4H_E_1_2_grade_3', 'H09-15863_A2H_E_1_10_grade_3', 'H09-15863_A2H_E_1_4_grade_3', 'H09-00804_A2H_E_1_4_grade_3', 'H09-18586_A2H_E_1_2_grade_3'
]

}

# GROUND_TRUTH = {
#     1:[
# 'Patient_011_02_grade_1', 'Patient_011_03_grade_1', 'Patient_013_02_grade_1', 'Patient_012_01_grade_1', 'Patient_013_01_grade_1', 'Patient_016_01_grade_1', 'Patient_015_02_grade_1', 'Patient_014_02_grade_1', 'Patient_016_02_grade_1', 'Patient_017_05_grade_1', 'Patient_013_04_grade_1', 'Patient_017_01_grade_1', 'Patient_014_01_grade_1', 'Patient_015_01_grade_1', 'Patient_011_04_grade_1', 'Patient_013_05_grade_1', 'Patient_015_03_grade_1', 'Patient_013_03_grade_1', 'Patient_015_04_grade_1', 'Patient_011_01_grade_1', 'Patient_017_03_grade_1', 'Patient_015_05_grade_1', 'Patient_017_04_grade_1', 'Patient_017_02_grade_1',
# 'Patient_005_01_grade_2', 'Patient_004_03_grade_2', 'Patient_002_01_grade_2', 'Patient_001_01_grade_2', 'Patient_001_03_grade_2', 'Patient_004_02_grade_2', 'Patient_003_02_grade_2', 'Patient_004_01_grade_2', 'Patient_001_02_grade_2', 'Patient_003_01_grade_2', 'Patient_005_02_grade_2',
# 'Patient_027_01_grade_3', 'Patient_027_03_grade_3', 'Patient_027_02_grade_3', 'Patient_027_06_grade_3', 'Patient_032_03_grade_3', 'Patient_032_01_grade_3', 'Patient_009_02_grade_3', 'Patient_032_04_grade_3', 'Patient_027_04_grade_3', 'Patient_022_01_grade_3', 'Patient_032_05_grade_3', 'Patient_022_04_grade_3',
#     ],
#
#     2:[
# 'Patient_020_03_grade_1', 'Patient_027_05_grade_1', 'Patient_021_03_grade_1', 'Patient_021_07_grade_1', 'Patient_020_01_grade_1', 'Patient_017_10_grade_1', 'Patient_018_02_grade_1', 'Patient_021_02_grade_1', 'Patient_017_11_grade_1', 'Patient_022_02_grade_1', 'Patient_019_04_grade_1', 'Patient_017_06_grade_1', 'Patient_017_09_grade_1', 'Patient_021_06_grade_1', 'Patient_017_07_grade_1', 'Patient_019_01_grade_1', 'Patient_019_05_grade_1', 'Patient_017_08_grade_1', 'Patient_018_01_grade_1', 'Patient_021_01_grade_1', 'Patient_019_03_grade_1', 'Patient_018_03_grade_1', 'Patient_019_02_grade_1',
# 'Patient_007_01_grade_2', 'Patient_006_02_grade_2', 'Patient_019_06_grade_2', 'Patient_009_01_grade_2', 'Patient_010_02_grade_2', 'Patient_008_02_grade_2', 'Patient_019_07_grade_2', 'Patient_005_03_grade_2', 'Patient_010_01_grade_2', 'Patient_008_01_grade_2', 'Patient_006_01_grade_2',
# 'Patient_032_09_grade_3', 'Patient_035_04_grade_3', 'Patient_035_06_grade_3', 'Patient_032_08_grade_3', 'Patient_032_06_grade_3', 'Patient_034_01_grade_3', 'Patient_035_08_grade_3', 'Patient_034_02_grade_3', 'Patient_032_10_grade_3', 'Patient_032_07_grade_3', 'Patient_035_05_grade_3', 'Patient_035_09_grade_3',
# ],
#   3:[
# 'Patient_036_03_grade_1', 'Patient_030_02_grade_1', 'Patient_031_03_grade_1', 'Patient_035_03_grade_1', 'Patient_033_01_grade_1', 'Patient_035_07_grade_1', 'Patient_033_03_grade_1', 'Patient_036_01_grade_1', 'Patient_029_01_grade_1', 'Patient_028_01_grade_1', 'Patient_029_02_grade_1', 'Patient_035_12_grade_1', 'Patient_032_02_grade_1', 'Patient_035_01_grade_1', 'Patient_038_01_grade_1', 'Patient_030_01_grade_1', 'Patient_038_02_grade_1', 'Patient_030_03_grade_1', 'Patient_033_02_grade_1', 'Patient_028_02_grade_1', 'Patient_036_02_grade_1', 'Patient_035_02_grade_1', 'Patient_031_02_grade_1', 'Patient_031_01_grade_1',
# 'Patient_020_02_grade_2', 'Patient_022_03_grade_2', 'Patient_025_02_grade_2', 'Patient_023_03_grade_2', 'Patient_021_04_grade_2', 'Patient_025_01_grade_2', 'Patient_021_05_grade_2', 'Patient_023_01_grade_2', 'Patient_023_02_grade_2', 'Patient_026_01_grade_2', 'Patient_024_01_grade_2',
# 'Patient_036_04_grade_3', 'Patient_037_06_grade_3', 'Patient_037_03_grade_3', 'Patient_036_05_grade_3', 'Patient_037_05_grade_3', 'Patient_037_02_grade_3', 'Patient_037_01_grade_3', 'Patient_035_10_grade_3', 'Patient_037_04_grade_3', 'Patient_035_13_grade_3', 'Patient_035_11_grade_3',
# ]
#
# }
class ImgLevelResult(object):
    def __init__(self, args):
        self.args = args
        self.imglist = {}
        for name in GROUND_TRUTH[CROSS_VAL[args.cross_val]]:
            # get the ground truth label: noral-0, low-level:1, high-level:2
            self.imglist[name.split('_grade')[0]] = int(name.split('_')[-1])-1
        self.prediction = defaultdict(list)

    def patch_result(self, name, label):
        name = name.split('/')[-1].split('_grade')[0]
        self.prediction[name].append(label)

    def batch_patch_result(self, names, labels):
        for name, label in zip(names, labels):
            name = name.split('/')[-1].split('_grade')[0]
            self.prediction[name].append(label)

    def final_result(self):
        result = []
        gt = []
        binary_result = []
        binary_gt = []
        for key, value in self.prediction.items():
            gt.append(self.imglist[key])
            binary_gt.append(1 if self.imglist[key]>0 else 0)
            _result = [value.count(0), value.count(1), value.count(2)]
            result.append(np.argmax(_result))
            binary_result.append(1 if np.argmax(_result)>0 else 0)
        acc = metrics.accuracy_score(gt, result)
        binary_acc = metrics.accuracy_score(binary_gt,binary_result)
        return acc,binary_acc