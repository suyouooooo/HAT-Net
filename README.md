# A Efficient Graph-based Framework for Multi-Organ Histology Image Classification
Code of Paper: **A Efficient Graph-based Framework for Multi-Organ Histology Image Classification**

Extended version of our BMVC 2021 paper: **HAT-Net: A Hierarchical Transformer Graph Neural Network for Grading of Colorectal Cancer Histology Images**
![alt text](images/p1.jpeg)

Our code is partially inspired by [CGC-Net](https://github.com/SIAAAAAA/CGC-Net), shout out to them.

## Requirements
Some of our installed packages:
- torch-geometric  1.7.2
- torch            1.8.2
- scikit-image     0.17.2
- scikit-learn     1.0
- opencv-python    4.5.3.56
- numpy            1.20.3
- matplotlib       3.4.2

## Datasets:

### 4 public histlogy image classification dataset:
- [CRC dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/crc_grading/): a colorectal cancer grading dataset:
- [Extended CRC dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/extended_crc_grading/): a colorectal cancer grading dataset.
- [UZH dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP): a prostate cancer grading dataset.
- [BACH dataset](https://zenodo.org/record/3632035#.YYzIq05ByUk): a breast cancer classification dataset.


## Train the model

<!-- single gpu -->
```
python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=15 --jk  --cv 3  --num_eval 5 --visualization --epochs 20   --stage 2 3 --depth 6
```
<!-- multi gpu -->


tested:
```
python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=15 --jk  --cv 3  --num_eval 5 --visualization --lr 0.0003 --epochs 20   --gamma 0.33 --stage 2 3 --depth 1 --load_data_list
```
or(has not been tested):
```
python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=15 --jk  --cv 3  --num_eval 5 --visualization --lr 0.0003 --epochs 20   --gamma 0.33 --stage 2 3 --depth 1
```

eval:
```
python -u eval.py  --norm_adj --batch-size=10 --step_size=10 --jk  --depth 1  --stage  2 3 --cv 1  --weight_file /data/by/tmp/HGIN/output/result/nuclei_soft-assign_l3x1_ar10_h20_o20_fca_%1fuse_adj0.4_sr1_d0.2_jkknn_cv2/Monday_24_May_2021_09h_50m_38s/model_best.pth.tar
```

visulization:
```
python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=15 --jk  --cv 3  --num_eval 5 --visualization --lr 0.0003 --epochs 20   --gamma 0.33 --stage 2 3 --depth 1
```


## Citation
If you used our code for your research, please cite:
```
@inproceedings{su2021,
  title="HAT-Net: A Hierarchical Transformer Graph Neural Network for Grading of Colorectal Cancer Histology Images",
  author={Yihan {Su} and Yu {Bai} and Bo {Zhang} and Zheng {Zhang} and Wendong {Wang}},
  booktitle={The British Machine Vision Conference (BMVC)},
  year={2021}
}
```