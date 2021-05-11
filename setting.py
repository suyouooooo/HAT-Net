import os
class CrossValidSetting:
    def __init__(self):
        self.name = 'CRC'
        self.batch_size = 16
        self.do_eval = False
        self.sample_time = 1
        self.sample_ratio = 1
        self.root = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning'#'/research/dept6/ynzhou/gcnn/data'
        self.save_path = '/home/suyihan/PycharmProjects/CGC-Net/output/AGS'
        self.log_path = os.path.join(self.save_path,'log' )
        self.result_path = os.path.join(self.save_path, 'result')
        self.dataset = 'CRC'
        self.max_edge_distance = 100
        self.max_num_nodes = 11404 # the maximum number of nodes in one graph
