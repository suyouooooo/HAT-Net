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