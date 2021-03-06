#export CUDA_VISIBLE_DEVICES=1 && python train_HGP_GCN.py --batch_size 10
export CUDA_VISIBLE_DEVICES=2
#echo $CUDA_VISIBLE_DEVICES
#python train_GIN_Hierarchical.py --load_data_list --norm_adj --batch-size=10 --step_size=10 --jk  --depth 6  --stage  2
#COMMAND="nohup bash run1.sh  &> ecrc_20epoch &"
#COMMAND="nohup bash run1.sh  &> hand_crafted_cv1_cv2_cv3&"
COMMAND="nohup bash run1.sh  &> drop10_30_crc_cv1_pan &"


python run1.py   --num_gpus  3   --command  "${COMMAND}"  --exec_gpu_id 2  --sleep_time 30
