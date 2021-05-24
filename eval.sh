#!/bin/bash
#export CUDA_VISIBLE_DEVICES=1 && python train_HGP_GCN.py --batch_size 10
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=$CUDA_HOME:/home/syh/cuda-11.0
export PATH=$PATH:/home/syh/cuda-11.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/syh/cuda-11.0/lib64

source /home/syh/anaconda3/bin/activate
#which conda
which python
#conda activate CGC-Net2
conda activate CGC-Net2-tensorboard

#python dataflow/construct_feature_graph.py

#python dataflow/consep.py

#python model/resnet.py

#python -u test_dataset.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --cv 1
#python -u test_dataset.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2
#python -u test_dataset.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --cv 3
#python -u test_dataset.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk
#python -u test_dataset.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk
#python -u test_dataset.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk
#python -u  /data/by/tmp/HGIN/dataflow/construct_feature_graph.py
#python -u /data/by/tmp/HGIN/dataflow/CNN_extract_GPU.py
#python -u  /data/by/tmp/HGIN/dataflow/
#python -u
#conda env list



#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 1 --load_data_sparse --num_eval 5
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2 --load_data_sparse --num_eval 5 --load_data_list
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 3 --load_data_sparse --num_eval 5

#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 1
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 3


#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=10 --step_size=10 --jk  --cv 3

#python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=10 --step_size=10 --jk  --depth 1  --stage  2 3 --cv 1  --num_eval 5 --visualization
python -u eval.py  --norm_adj --batch-size=10 --step_size=10 --jk  --depth 1  --stage  2 3 --cv 1  --weight_file /data/by/tmp/HGIN/output/result/nuclei_soft-assign_l3x1_ar10_h20_o20_fca_%1fuse_adj0.4_sr1_d0.2_jkknn_cv2/Monday_24_May_2021_09h_50m_38s/model_best.pth.tar
#python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=10 --jk  --depth 1  --stage  2 3 --cv 2  --num_eval 5
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --depth 1  --stage  2 3 --cv 3  --num_eval 5

#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --depth 6  --stage  2 3 --cv 1 --num_eval 5 --load_data_sparse
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --depth 6  --stage  2 3 --cv 2 --num_eval 5 --load_data_sparse
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --depth 6  --stage  2 3 --cv 3 --num_eval 5 --load_data_sparse

#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --depth 1  --stage  2 3 --cv 1
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --depth 1  --stage  2 3 --cv 2
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --depth 1  --stage  2 3 --cv 3




#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --depth 6  --stage  2
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --depth 6  --stage  2
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --depth 6  --stage  2
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=10 --step_size=20 --jk   --lr 0.001
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=10 --step_size=10 --jk  --depth 6  --stage  2
