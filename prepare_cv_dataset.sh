#!/bin/bash
#export CUDA_VISIBLE_DEVICES=1 && python train_HGP_GCN.py --batch_size 10
export CUDA_VISIBLE_DEVICES=3
#export CUDA_HOME=$CUDA_HOME:/home/syh/cuda-11.0
#export PATH=$PATH:/home/syh/cuda-11.0/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/syh/cuda-11.0/lib64

source /home/suyihan/anaconda3/bin/activate
#which conda
conda activate CGC-Net_by

which python
#conda activate CGC-Net2-tensorboard
#cd /data/by/tmp/HGIN

#python /data/by/tmp/HGIN/dataflow/prepare_cv_dataset_no_random.py
#python /data/by/tmp/HGIN/HGIN/common/utils.py
#python  dataflow/construct_training_data.py
python  /home/baiyu/HGIN/dataflow/file_dataloader.py

#python -u train_consep.py -net resnet50 -b 512 -lr 0.1 -gpu -e 200
#python -u train_consep.py -net resnet50 -b 512 -lr 0.1 -gpu -e 200
#python -u eval_consep.py -net resnet50 -b 512  -gpu  -weights /data/by/tmp/HGIN/checkpoint/resnet50/Monday_31_May_2021_00h_40m_26s/17-best.pth
#python -u eval_consep.py -net resnet50 -b 512  -gpu  -weights /data/by/tmp/HGIN/checkpoint/resnet50/Tuesday_01_June_2021_19h_43m_05s/191-best.pth
#python -u dataflow/construct_feature_graph_cnn.py
#python -u /data/by/tmp/HGIN/dataflow/prepare_cv_dataset.py
#python -u dataflow/construct_feature_graph.py

#python train_consep.py -net resnet50 -b 512 -lr 0.1 -gpu -e 200
#python train_consep.py -net resnet50 -b 512 -lr 0.1 -gpu -e 200
#python dataflow/construct_feature_graph.py

#python dataflow/consep.py

#python model/resnet.py

#python -u test_dataset.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --cv 1 --stage 2 3 --depth 6
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
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 3 --load_data_sparse --num_eval 5  --epochs 20 --stage 2 3 --depth 1
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 3 --load_data_sparse --num_eval 5  --epochs 20 --stage 2 3 --depth 1
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 3 --load_data_sparse --num_eval 5  --epochs 20 --stage 2 3 --depth 1

#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2 --load_data_sparse --num_eval 5  --epochs 20 --stage 2 3 --depth 6 --num_workers 4
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 1 --load_data_sparse --num_eval 5  --epochs 20 --stage 2 3 --depth 6 --num_workers 4
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 3 --load_data_sparse --num_eval 5  --epochs 20 --stage 2 3 --depth 6 --num_workers 4

#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2 --load_data_sparse --num_eval 5  --epochs 20 --stage 2 3 --depth 6
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2 --load_data_sparse --num_eval 5  --epochs 20 --stage 2 3 --depth 6
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2 --load_data_sparse --num_eval 5  --epochs 20 --stage 2 3 --depth 6
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 1 --load_data_sparse --num_eval 5  --epochs 20 --stage 2 3 --depth 6
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2 --load_data_sparse --num_eval 5  --epochs 20 --stage 2 3 --depth 6
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2 --load_data_sparse --num_eval 5  --epochs 20 --stage 2 3 --depth 1
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2 --load_data_sparse --num_eval 5  --epochs 20 --stage 2 3 --depth 1
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2 --load_data_sparse --num_eval 5  --epochs 20 --stage 2 3 --depth 1

#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 1
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --cv 3


#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=10 --step_size=10 --jk  --cv 3

#python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=10 --jk  --depth 1  --stage  2 3 --cv 3  --num_eval 5 --visualization --lr 0.0002
#python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=15 --jk  --cv 2  --num_eval 5 --visualization --lr 0.0003 --epochs 20  --step_size 15 --gamma 0.33
#python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=15 --jk  --cv 2  --num_eval 5 --visualization --lr 0.0003 --epochs 20  --step_size 15 --gamma 0.33

# s 23 d 1  cv2
#python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=10 --jk  --cv 3  --num_eval 5 --visualization --lr 0.001 --epochs 15  --gamma 0.1 --stage 2 3 --depth 6
#python -u train_GIN_Hierarchical.py --visualization --norm_adj --batch-size=20 --step_size=10 --jk  --cv=3  --num_eval=5  --depth=6  --stage 2 3 --epochs=15 --name avg
#python -u train_GIN_Hierarchical.py --visualization --norm_adj --batch-size=20 --step_size=10 --jk  --cv=2  --num_eval=5  --depth=6  --stage 2 3 --epochs=15 --name avg
#python -u train_GIN_Hierarchical.py --visualization --norm_adj --batch-size=20 --step_size=10 --jk  --cv=1  --num_eval=5  --depth=6  --stage 2 3 --epochs=15 --name avg
#python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=10 --jk  --cv 1  --num_eval 5 --visualization --lr 0.001 --epochs 15  --stage 2 3 --depth 6
#python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2  --num_eval 5 --visualization --lr 0.001 --epochs 15  --stage 2 3 --depth 6
#python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=10 --jk  --cv 3  --num_eval 5 --visualization --lr 0.001 --epochs 15  --stage 2 3 --depth 6

#python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=10 --jk  --cv 1  --num_eval 5 --visualization --lr 0.001 --epochs 30   --gamma 0.1 --stage 2 --depth 6
#python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=10 --jk  --cv 2  --num_eval 5 --visualization --lr 0.001 --epochs 30   --gamma 0.1 --stage 2 --depth 6
#python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=10 --jk  --cv 3  --num_eval 5 --visualization --lr 0.001 --epochs 30   --gamma 0.1 --stage 2 --depth 6

# s 23 d 6  cv3
#python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=15 --jk  --cv 3  --num_eval 5 --visualization --lr 0.0003 --epochs 20   --gamma 0.33  --stage 2 3 --depth 6

#python -u train_GIN_Hierarchical.py  --norm_adj --batch-size=20 --step_size=10 --jk  --depth 1  --stage  2 3 --cv 2  --num_eval 5
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --depth 1  --stage  2 3 --cv 3  --num_eval 5

#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --depth 6  --stage  2 3 --cv 1 --num_eval 5 --load_data_sparse
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --depth 6  --stage  2 3 --cv 2 --num_eval 5 --load_data_sparse
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk  --depth 6  --stage  2 3 --cv 3 --num_eval 5 --load_data_sparse

#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --depth 1  --stage  2 3 --cv 1
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --depth 1  --stage  2 3 --cv 2
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --depth 1  --stage  2 3 --cv 3




#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk --cv=1 --num_eval=5 --network=CGC --epochs=15 --num_workers 4
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk --cv=2 --num_eval=5 --network=CGC --epochs=15 --num_workers 4
#python -u train_GIN_Hierarchical.py --norm_adj --batch-size=20 --step_size=10 --jk --cv=3 --num_eval=5 --network=CGC --epochs=15 --num_workers 4
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --depth 6  --stage  2
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --depth 6  --stage  2
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=20 --step_size=10 --jk  --depth 6  --stage  2
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=10 --step_size=20 --jk   --lr 0.001
#python -u train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=10 --step_size=10 --jk  --depth 6  --stage  2
