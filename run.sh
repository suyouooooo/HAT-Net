#export CUDA_VISIBLE_DEVICES=1 && python train_HGP_GCN.py --batch_size 10
export CUDA_VISIBLE_DEVICES=1 
#/home/by/miniconda3/bin/conda activate 
/home/syh/anaconda3/envs/CGC-Net2/bin/python train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=10 --step_size=10 --jk
