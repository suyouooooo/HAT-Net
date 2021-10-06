import glob
from common.metric import ImgLevelResult
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import pdb
from tqdm import tqdm
import argparse
import os
import json
import time
from common.utils import mkdirs, save_checkpoint, load_checkpoint, init_optim, output_to_gexf, Metric
from torch.optim import lr_scheduler
from model import network_GIN_Hierarchical
from model.network_GIN_baiyu import HatNet
from torch_geometric.nn import DataParallel
from dataflow.data import prepare_train_val_loader, get_ecrc_dataset, get_bach_dataset
from setting import CrossValidSetting as DataSetting
import torch.utils.checkpoint as cp
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from datetime import datetime
from common.utils import (
    visualize_scalar,
    visualize_lastlayer,
    visualize_network,
    visualize_param_hist
)


DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

def write_csv(image_names, results):
    import csv
    import re
    header = ['case', 'class']
    #csv_fp = generate_csv_path(label_folder)
    csv_fp = 'pred.csv'
    #print(csv_fp)
    with open(csv_fp, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        count = 0
        for image_name, class_id in zip(image_names, results):
            #print(image_name)
            image_name = os.path.basename(image_name)
            #print(help(re.search))
            image_id = re.search(r'test([0-9]+)\.tif', image_name).group(1)
            #print(image_id)
            row = [image_id, class_id]
            if class_id != 0:
                print(row)

            #count += 1
            #print(row)
            writer.writerow(row)

def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    #metric = Metric()
    metric = Metric(num_classes=args.num_classes)
    model.eval()
    device = 'cuda:1' if torch.cuda.device_count()>1 else 'cuda:0'
    torch.cuda.empty_cache()
    finaleval = ImgLevelResult(args)
    image_names = []
    results = []
    with torch.no_grad():
        test_time = args.test_epoch if (args.dynamic_graph and name !='Train')else 1
        if args.visualization:
            test_time = 1
        pred_n_times = []
        labels_n_time = []

        #print(test_time)
        print(test_time)
        for _ in range(test_time):
            # test 5 times, each time the graph is constructed by the same method from that in train
            preds = []
            labels = []
            #dataset.dataset.set_val_epoch(_)

            for batch_idx, data in enumerate(dataset):

                if args.load_data_list:
                    patch_name = [dataset.dataset.idxlist[d.patch_idx.item()] for d in data]
                    label = torch.cat([d.y for d in data]).numpy()
                else:
                    patch_name = [dataset.dataset.idxlist[patch_idx.item()] for patch_idx in data.patch_idx]
                    data.to('cuda:0')
                    label = data.y.cpu().numpy()

                for p in patch_name:
                    print(p)
                    image_names.append(p.split('_grade_')[0] + '.tif')
                    #image_names.append(p.replace('_grade_')[0].replace('test', ''))
                print(image_names)

                ypred, _, = model(data)
                preds = torch.argmax(ypred, dim=-1)

                for r in preds:
                    r = r.cpu().item()
                    if r == 0:
                        #print(r)
                        r = 1
                        #print(r)
                        #print()
                    elif r == 1:
                        r = 0
                    #results.append(r.cpu().numpy())
                    results.append(r)
                #print(results)
                metric.update(ypred, torch.tensor(label), patch_name)

    write_csv(image_names, results)
    from zipfile import ZipFile

    with ZipFile('predictions.zip', 'w') as predictions:
        predictions.write('pred.csv')

    patch_acc = metric.patch_accuracy()
    image_acc_three = metric.image_acc_three_class()
    image_acc_bin = metric.image_acc_binary_class()
    kappa = metric.kappa()
    auc = metric.auc()

    result = {'patch_acc': patch_acc, 'img_acc':image_acc_three, 'binary_acc':  image_acc_bin, 'kappa': kappa, 'auc': auc}
    return result

def cell_graph(args, writer = None):
    # val==test loader since we do cross-val

    #train_loader, val_loader, test_loader = prepare_train_val_loader(args)
    #train_loader, val_loader, test_loader = get_ecrc_dataset(args)
    train_loader, val_loader, test_loader = get_bach_dataset(args)
    setting = DataSetting()
    input_dim = args.input_feature_dim
    if args.task == 'CRC':
        args.num_classes = 3
    elif args.task == 'ECRC':
        args.num_classes = 3
    elif args.task == 'TCGA':
        args.num_classes = 2
    elif args.task == 'BACH':
        args.num_classes = 4
    else:
        raise ValueError('wrong task name')
    # model = atten_network.SpGAT(18,args.hidden_dim,3, args.drop_out, args.assign_ratio,3)
    #model = network_GIN_Hierarchical.SoftPoolingGcnEncoder(setting.max_num_nodes,
    #    input_dim, args.hidden_dim, args.output_dim, True, True, args.hidden_dim,  args.num_classes,
    #                                      args.assign_ratio,[50], concat= True,
    #                                      gcn_name= args.gcn_name,collect_assign=args.visualization,
    #                                      #load_data_sparse=(args.load_data_list and not args.visualization),
    #                                      #load_data_sparse=args.load_data_sparse,
    #                                      #load_data_sparse=(not args.load_data_list),
    #                                      load_data_sparse=True,
    #                                      norm_adj=args.norm_adj, activation=args.activation, drop_out=args.drop_out,
    #                                      jk=args.jump_knowledge,
    #                                      depth=args.depth,
    #                                      stage=args.stage
    #                                      )
    model = HatNet(512, 64, args.num_classes)

    #for i in glob.
    model.load_state_dict(torch.load(args.weight_file)['state_dict'])

    model = model.cuda()
    #if torch.cuda.device_count() > 1 :
    #    print('use %d GPUs for training!'% torch.cuda.device_count())

    #    if args.load_data_list:
    #        model = DataParallel(model).cuda()
    #    else:
    #        model = nn.DataParallel(model).cuda()
    #else:
    #    if args.load_data_list and not args.visualization:
    #        model = DataParallel(model).cuda()
    #    else:
    #        model = model.cuda()

    val_result = evaluate(val_loader, model, args, name='Validation')
    print(val_result)

def arg_parse():
    data_setting = DataSetting()
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset',
            help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    softpool_parser = parser.add_argument_group()
    ## yihan
    softpool_parser.add_argument('--accumulation_steps', dest='accumulation_steps', type=int, default=2,
                                 help='accumulation_steps')
    ##
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float, default=0.10,
                                 help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
                                 help='number of pooling layers')
    parser.add_argument('--datadir', dest='datadir',
                        help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
                        help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
                        help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=4,
                        help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type', default='ca',
                        help='[c, ca, cal, cl] c: coor, a:appearance, l:soft-label')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
                        help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int, help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const', const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float, help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const', const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--sample-ratio', dest='sample_ratio', default=1, )
    parser.add_argument('--sample-time', dest='sample_time', default=1)
    parser.add_argument('--visualization', action='store_const', const=True,
                        help='use assignment matrix for visualization')
    parser.add_argument('--method', dest='method', help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix', help='suffix added to the output filename')
    parser.add_argument('--input_feature_dim', dest='input_feature_dim', type=int,
                        help='the feature number for each node', default=18)
    parser.add_argument('--resume', default=False, )
    parser.add_argument('--optim', dest='optimizer', help='name for the optimizer, [adam, sgd, rmsprop] ')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    parser.add_argument('--step_size', default=10, type=int, metavar='N',
                        help='stepsize to decay learning rate (>0 means this is enabled)')
    parser.add_argument('--skip_train', action='store_const',
                        const=True, default=False, help='only do evaluation')
    parser.add_argument('--normalize', default=False, help='normalize the adj matrix or not')
    #parser.add_argument('--load_data_list', action='store_true', default=True)
    #parser.add_argument('--load_data_sparse', action='store_true', default=False)
    parser.add_argument('--load_data_list', action='store_true')
    parser.add_argument('--load_data_sparse', action='store_true')
    parser.add_argument('--name', default='fuse')
    parser.add_argument('--gcn_name', default='SAGE')
    parser.add_argument('--active', dest='activation', default='relu')
    parser.add_argument('--dynamic_graph', dest='dynamic_graph', action='store_const', const=True, default=False, )
    parser.add_argument('--sampling_method', default='fuse', )
    parser.add_argument('--test_epoch', default=5, type=int)
    parser.add_argument('--sita', default=1., type=float)

    parser.add_argument('--norm_adj', action='store_const', const=True, default=False, )
    parser.add_argument('--readout', default='max', type=str)
    parser.add_argument('--task', default='CRC', type=str)
    parser.add_argument('--mask', default='cia', type=str)
    parser.add_argument('--n', dest='neighbour', default=8, type=int)
    parser.add_argument('--sample_ratio', default=0.5, type=float)
    parser.add_argument('--drop', dest='drop_out', default=0.2, type=float)
    parser.add_argument('--noise', dest='add_noise', action='store_const', const=True, default=False, )
    parser.add_argument('--valid_full', action='store_const', const=True, default=False, )
    parser.add_argument('--dist_g', dest='distance_prob_graph', action='store_const', const=True, default=False, )
    parser.add_argument('--jk', dest='jump_knowledge', action='store_const', const=True, default=True)
    parser.add_argument('--g', dest='graph_sampler', default='knn', type=str)
    parser.add_argument('--cv', dest='cross_val', default=1, type=int)

    parser.add_argument('--depth', default=None, type=int)
    parser.add_argument('--stage', nargs='*', type=int)
    parser.add_argument('--num_eval', default=1, type=int)
    parser.add_argument('--weight_file', required=True, type=str)

    parser.set_defaults(datadir=data_setting.root,
                        logdir=data_setting.log_path,
                        resultdir=data_setting.result_path,
                        sample_time=data_setting.sample_time,
                        dataset='nuclei',
                        max_nodes=16000,  # no use
                        cuda='0',
                        feature='ca',
                        lr=0.001,
                        clip=2.0,
                        batch_size=40,
                        num_epochs=30,
                        num_workers=2,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=3,
                        num_gc_layers=3,
                        dropout=0.2,
                        method='soft-assign',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1,
                        input_feature_dim=18,
                        optim='adam',
                        weight_decay=1e-4,
                        step_size=10,
                        gamma=0.1,
                        dynamic_graph=False,
                        test_epoch=5,

                        )
    return parser.parse_args()


def main():
    settings = DataSetting()
    prog_args = arg_parse()
    #torch.backends.cudnn.benchmark = True
    print(prog_args)
    writer = None
    cell_graph(prog_args, writer=writer)

if __name__ == "__main__":
    #torch.multiprocessing.set_sharing_strategy('file_system')
    print('shared strategy', torch.multiprocessing.get_sharing_strategy())
    main()
