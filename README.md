# HGIN


## train

single gpu
``
python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2  --num_eval 5
``
multi gpu

tested:
``
python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2  --num_eval 5 --load_data_list
``
or(has not been tested):
``
python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2  --num_eval 5
``

eval:
```
python -u eval.py  --norm_adj --batch-size=10 --step_size=10 --jk  --depth 1  --stage  2 3 --cv 1  --weight_file /data/by/tmp/HGIN/output/result/nuclei_soft-assign_l3x1_ar10_h20_o20_fca_%1fuse_adj0.4_sr1_d0.2_jkknn_cv2/Monday_24_May_2021_09h_50m_38s/model_best.pth.tar
```

visulization:
```
python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2  --num_eval 5 --visualization
```