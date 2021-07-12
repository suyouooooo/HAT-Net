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
#cd /data/by/tmp/HGIN

python -u /data/by/tmp/HGIN/compute_mean.py
