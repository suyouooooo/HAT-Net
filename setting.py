import os
from datetime import datetime
class CrossValidSetting:
    def __init__(self):
        self.name = 'CRC'
        self.batch_size = 16
        self.do_eval = False
        self.sample_time = 1
        self.sample_ratio = 1
        # processed
        #self.root = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning'#'/research/dept6/ynzhou/gcnn/data'
        self.root = '/data/smb/syh/PycharmProjects/CGC-Net/data_res50/'#'/research/dept6/ynzhou/gcnn/data'
        #self.root = '/home/baiyu/Extended_CRC_Graph'#'/research/dept6/ynzhou/gcnn/data'
        self.save_path = 'output'
        self.log_path = os.path.join(self.save_path,'log' )
        self.result_path = os.path.join(self.save_path, 'result')
        self.dataset = 'CRC'
        self.max_edge_distance = 100
        self.max_num_nodes = 11404 # the maximum number of nodes in one graph
        self.log_dir = 'runs'


class ConSepSettings:
    def __init__(self):
        #self.train_root = '/data/by/tmp/HGIN/dataflow/consep_data/train/Images'
        self.train_root = '/data/by/tmp/HGIN/dataflow/consep_data/train.txt'
        self.test_root = '/data/by/tmp/HGIN/dataflow/consep_data/test.txt'
        self.checkpoint = 'checkpoint'
        self.log_path = 'runs'
        # self.epochs = 200
        self.milestones = [60, 120, 160]
        DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
        self.timenow = datetime.now().strftime(DATE_FORMAT)
        self.save_epoch = 10
        self.image_size = 64
