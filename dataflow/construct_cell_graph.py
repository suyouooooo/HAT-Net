import json
import os
from pathlib import Path
from functools import partial
import sys
import random
import pickle
import glob
import string
import time
#sys.path.append('../')
sys.path.append(os.getcwd())

import lmdb
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import adaptive_avg_pool3d
from torchvision import transforms
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_cluster import grid_cluster
from torch_scatter import scatter
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos


import cv2
import numpy as np
from skimage.measure import regionprops


from stich import JsonFolder, JsonDataset, ImageFolder, ImageDataset





def feature_extract_resnet50(image, bboxes, extract_func):
    """
        bboxes: (bboxes :bbox [min_y, min_x, max_y, max_x])
    """
