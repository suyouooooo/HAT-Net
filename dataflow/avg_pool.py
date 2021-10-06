

import glob
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.nn.functional import adaptive_avg_pool1d
from torch_geometric.data import DataLoader



def avg_pool(data, output_dim):
    data.x = adaptive_avg_pool1d(data.x[None, :, :], (output_dim)).squeeze()
    return data

class PtReader:
    def __init__(self, path, out_dim):
        search_path = os.path.join(path, '**', '*.pt')
        self.pt_names = glob.glob(search_path, recursive=True)
        self.out_dim = out_dim

    def __len__(self):
        return len(self.pt_names)

    def __getitem__(self, idx):
        pt_path = self.pt_names[idx]
        data = torch.load(pt_path)
        data = avg_pool(data, self.out_dim)
        return data

class PtGraphWriter:
    def __init__(self, save_path):
        self.save_path = save_path
        self.thread_pool = ThreadPoolExecutor(8)

    def write_sample(self, data):

        basename = os.path.basename(data.path)
        save_fp = os.path.join(self.save_path, basename)
        print(save_fp, data.x.shape)
        #save_path = os.path.join(self.save_path, os.path.basename(data.path.replace('.jpg', '.pt')))
        #print(111,  save_path)
        #print(data)
        #sys.exit()
        if len(data.x) <= 5:
            return

        #print(save_path)
        #print(save_path)
        torch.save(data.clone(), save_fp)

    def write_batch(self, batch):
        self.thread_pool.map(
            self.write_sample,
            [batch[i] for i in range(batch.num_graphs)]
        )

def main():
    #src_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/test/hatnet2048dim/'
    #dest_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/test/hatnet512dim'
    src_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/Aug/hatnet2048dim/'
    #dest_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/test/hatnet512dim'
    #dest_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/Aug/hatnet512dim'
    #dest_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/Aug/hatnet128dim'
    #dest_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/test/hatnet128dim'
    #dest_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/Aug/hatnet32dim'
    #dest_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/test/hatnet32dim'
    dest_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Cell_Graph/Aug/hatnet32dim'
    output_dim = 32

    cg_reader = PtReader(src_path, output_dim)
    cg_writter = PtGraphWriter(dest_path)

    data_loader = DataLoader(cg_reader, num_workers=4, batch_size=16)


    for batch in data_loader:
        cg_writter.write_batch(batch)

    #for i in glob.iglob(os.path.join(src_path, '**', '*.pt'), recursive=True):
        #data = torch.load(i)
        #print(save_fp)



if __name__ == '__main__':
    main()