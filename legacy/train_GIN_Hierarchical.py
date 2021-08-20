from common.metric import ImgLevelResult
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from model import network_GIN_Hierarchical, network_CGCNet
from torch_geometric.nn import DataParallel
from dataflow.data import prepare_train_val_loader
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

def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    metric = Metric()
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        print('eval......')
        for batch_idx, data in enumerate(dataset):

            if args.load_data_list:
                patch_name = [dataset.dataset.idxlist[d.patch_idx.item()] for d in data]
                label = torch.cat([d.y for d in data]).numpy()
            else:
                print('else......')
                patch_name = [dataset.dataset.idxlist[patch_idx.item()] for patch_idx in data.patch_idx]
                label = data.y.cpu().numpy()
                #data = data.cuda()
                print('to cuda')
                data.to('cuda:0')


                print('model')
                ypred = model(data)
                pred = torch.argmax(ypred, dim=-1)
                print('update')
                metric.update(pred, torch.tensor(label), patch_name)


    print('here........')
    patch_acc = metric.patch_accuracy()
    image_acc_three = metric.image_acc_three_class()
    image_acc_bin = metric.image_acc_binary_class()
    print('stuck........')

    #multi_class_acc,binary_acc = finaleval.final_result()
    #result = {'patch_acc': metrics.accuracy_score(labels_n_time,pred_n_times), 'img_acc':multi_class_acc, 'binary_acc': binary_acc }
    result = {'patch_acc': patch_acc, 'img_acc':image_acc_three, 'binary_acc':  image_acc_bin}
    return result

def gen_prefix(args):

    name = args.dataset
    name += '_' + args.method
    if args.method == 'soft-assign':
        name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
        name += '_ar' + str(int(args.assign_ratio*100))

    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    name +=  '_f' +args.feature_type
    name += '_%' + str(args.sample_ratio)
    # name += '_' + args.sample_method
    name += '_name' + args.name

    if args.load_data_sparse:
        name += '_sp'
    if args.load_data_list:
        name +='_list'
    if args.norm_adj:
        name+='_adj0.4'
    if args.activation !='relu':
        name+=args.activation
    if args.readout =='mix':
        name+=args.readout
    if args.task != 'CRC':
        name+=('_'+args.task)
    if args.mask !='cia':
        name +='hvnet'
    if args.neighbour !=8:
        name +='_n'+str(args.neighbour)
    name += '_sr' + str(args.sample_ratio)
    if args.drop_out >0:
        name +='_d' + str(args.drop_out)
    if args.jump_knowledge:
        name +='_jk'
    name += args.graph_sampler
    if args.cross_val:
        name +='_cv'+str(args.cross_val)
    if args.stage:
        name += '_stage'
        for s in args.stage:
            name += str(s)
    if args.depth:
        name += '_depth'
        name += str(args.depth)
    if args.num_epochs:
        name += '_epochs'
        name += str(args.num_epochs)
    if args.lr:
        name += '_lr'
        name += str(args.lr)
    if args.network:
        name += '_network'
        name += str(args.network)
    if args.gamma:
        name += '_gamma'
        name += str(args.gamma)
    print('name', name)

    return name

def eval_idx(total_iters, num_evals):
    interval = total_iters // num_evals
    first_interval = total_iters - interval * num_evals + interval - 1
    intervals = []
    intervals.append(first_interval)
    for i in range(num_evals - 1):
        intervals.append(interval + intervals[-1])

    return intervals

def max_grad(model):
    res = 0
    max_name = ''
    for name, p in model.named_parameters():
        cur_max = p.grad.max()
        if res < cur_max:
            res = cur_max
            max_name = name

    print(name, res)
    return res

def train(dataset, model, args,  val_dataset=None, test_dataset=None, writer=None, checkpoint = None):
    print('train data loader type', type(dataset))
    print('val data loader type', type(val_dataset))
    print('model type', type(model))
    print("==> Start training")
    device = 'cuda:1' if torch.cuda.device_count()>1 else 'cuda:0'
    start_epoch = 0
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
    #print(optimizer)
    #sys.exit()
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    if args.step_size > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    save_path = os.path.join(args.resultdir, gen_prefix(args), TIME_NOW)
    best_val_result = {'patch_acc': 0, 'img_acc': 0, 'binary_acc': 0}


    eval_idxes = eval_idx(len(dataset), args.num_eval)
    if args.visualization:
        eval_count = 0
    for epoch in range(start_epoch, args.num_epochs):
        epoch_start = time.time()
        model.train()
        if args.name == 'fuse':
            dataset.dataset.set_epoch(epoch)

        for batch_idx, data in enumerate(dataset):
            if not args.load_data_list:
                #data = data.cuda()
                data.to('cuda:0')

            _, loss = model(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            max_grad(model)

            print('Training Loss:{:0.6f}, Epoch: {epoch}, Batch: [{batch_idx}/{total}] LR:{:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                batch_idx=batch_idx,
                total=len(dataset) - 1
            ))

            if args.visualization:
                n_iter = epoch *  len(dataset) + batch_idx + 1
                visualize_lastlayer(writer, model, n_iter)
                visualize_scalar(writer, 'Train/loss', loss.item(), n_iter)
                visualize_scalar(writer, 'Train/lr', optimizer.param_groups[0]['lr'], n_iter)

            if batch_idx in eval_idxes:
                print(val_dataset)
                eval_start = time.time()
                print('Evaluating at {}th iterations.......'.format(batch_idx))
                val_result = evaluate(val_dataset, model, args, name='Validation')
                if val_result['img_acc'] > best_val_result['img_acc']:
                    best_val_result['img_acc'] =  val_result['img_acc']

                    print('Saving best weight file to {}'.format(save_path))
                    save_checkpoint({'epoch': epoch + 1,
                                     'state_dict': model.state_dict() if torch.cuda.device_count() < 2 else model.module.state_dict(),
                                     'optimizer': optimizer.state_dict(),
                                     'val_acc': val_result},
                                    os.path.join(save_path, 'model_best.pth.tar'))

                if val_result['patch_acc'] > best_val_result['patch_acc']:
                    best_val_result['patch_acc'] = val_result['patch_acc']

                if val_result['binary_acc'] > best_val_result['binary_acc']:
                    best_val_result['binary_acc'] = val_result['binary_acc']

                print(('poch: {}, eval time consumed: {:0.4f}s, val patch acc: {:0.6f}, val image acc: {:0.6f}, val binary acc: {:0.6f}, '
                        'best val patch acc: {:0.6f}, best val image acc: {:0.6f}, best val binary acc: {:0.6f}').format(
                        epoch,
                        time.time() - eval_start,
                        val_result['patch_acc'],
                        val_result['img_acc'],
                        val_result['binary_acc'],
                        best_val_result['patch_acc'],
                        best_val_result['img_acc'],
                        best_val_result['binary_acc'],
                    ))
                print()
                if args.visualization:
                    eval_count += 1
                    visualize_scalar(writer, 'Val/patch_acc', val_result['patch_acc'],  eval_count)
                    visualize_scalar(writer, 'Val/image_acc', val_result['img_acc'],  eval_count)
                    visualize_scalar(writer, 'Val/binary_acc', val_result['binary_acc'],  eval_count)

                model.train()

        if args.step_size > 0:
            scheduler.step()

        print('training time consumed:{:2f}s'.format(
            time.time() - epoch_start
        ))


def cell_graph(args, writer = None):
    # val==test loader since we do cross-val

    train_loader, val_loader, test_loader = prepare_train_val_loader(args)
    setting = DataSetting()
    input_dim = args.input_feature_dim
    if args.task == 'CRC':
        args.num_classes = 3
    elif args.task == 'ECRC':
        args.num_classes = 3
    else:
        raise ValueError('wrong task name')
    # model = atten_network.SpGAT(18,args.hidden_dim,3, args.drop_out, args.assign_ratio,3)
    if args.network == 'HGTIN':
        model = network_GIN_Hierarchical.SoftPoolingGcnEncoder(setting.max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, True, True, args.hidden_dim,  args.num_classes,
                                              args.assign_ratio,[50], concat= True,
                                              gcn_name= args.gcn_name,collect_assign=args.visualization,
                                              #load_data_sparse=(args.load_data_list and not args.visualization),
                                              #load_data_sparse=args.load_data_sparse,
                                              #load_data_sparse=(not args.load_data_list),
                                              load_data_sparse=True,
                                              norm_adj=args.norm_adj, activation=args.activation, drop_out=args.drop_out,
                                              jk=args.jump_knowledge,
                                              depth=args.depth,
                                              stage=args.stage,
                                              jk_tec=args.jk_tec,
                                              pool_tec=args.pool_tec
                                              )
    elif args.network == 'CGC':
        model = network_CGCNet.SoftPoolingGcnEncoder(setting.max_num_nodes,
                                              input_dim, args.hidden_dim, args.output_dim, True, True, args.hidden_dim,
                                              args.num_classes,
                                              args.assign_ratio, [50], concat=True,
                                              gcn_name=args.gcn_name, collect_assign=args.visualization,
                                              load_data_sparse=True,
                                              norm_adj=args.norm_adj, activation=args.activation,
                                              drop_out=args.drop_out,
                                              jk=args.jump_knowledge
                                              )

    #print(model)
    #if args.cross_val == 1:
    #    model_path = '/home/baiyu/HGIN/output/result/nuclei_soft-assign_l3x1_ar10_h20_o20_fca_%1_nameavg_adj0.4_ECRC_sr1_d0.2_jkknn_cv1_stage23_depth6_epochs35_lr0.001_networkHGTIN_gamma0.1/Wednesday_28_July_2021_20h_49m_55s/model_best.pth.tar'
    #    print('loading file from {}'.format(model_path))
    #    model.load_state_dict(torch.load(model_path)['state_dict'])
    #    print('done')

    #tensor = torch.Tensor(3, 10, 16)
    if(args.resume):
        if args.resume == 'best':
            resume_file = 'model_best.pth.tar'
            resume_path = os.path.join(args.resultdir, gen_prefix(args), resume_file)
        elif args.resume == 'weight':
            resume_file = 'weight.pth.tar'
            resume_path = os.path.join(args.resultdir, gen_prefix(args), resume_file)
        else:#'/media/amanda/HDD2T_1/warwick-research/experiment/gcnn/result'
            resume_path  =  os.path.join(args.resultdir,args.resume,'model_best.pth.tar')
        checkpoint = load_checkpoint(resume_path)
        model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.device_count() > 1 :
        print('use %d GPUs for training!'% torch.cuda.device_count())

        if args.load_data_list:
            model = DataParallel(model).cuda()
        else:
            model = nn.DataParallel(model).cuda()
    else:
        if args.load_data_list and not args.visualization:
            model = DataParallel(model).cuda()
        else:
            model = model.cuda()
    #print(type(model))
    if not args.skip_train:
        # 如果不跳过训练，就执行下面的操作
        if args.resume:
            train(train_loader, model, args, val_dataset=val_loader, test_dataset=None, writer=writer, checkpoint = checkpoint)
        else:
            train(train_loader, model, args, val_dataset=val_loader, test_dataset=None, writer=writer)
        #print('finally: max_val_acc:%f'%max(val_accs))
    #_ = evaluate(test_loader, model, args, name='Validation', max_num_examples=None)

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

    parser.add_argument('--network', default='HGTIN', type=str)
    parser.add_argument('--jk_tec', default='lstm', type=str)
    parser.add_argument('--pool_tec', default='mincut', type=str)

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
    #print(gen_prefix(prog_args))
    #import sys
    #sys.exit()
    writer = None
    if prog_args.visualization:
        tb_logdir = os.path.join(settings.log_dir, gen_prefix(prog_args), TIME_NOW)
        print(tb_logdir)
        mkdirs([tb_logdir])
        writer = SummaryWriter(log_dir=tb_logdir)
    #if os.path.exists()
    #tb_logdir = os.path.join()
    cell_graph(prog_args, writer=writer)

if __name__ == "__main__":
    #torch.multiprocessing.set_sharing_strategy('file_system')

    print('shared strategy', torch.multiprocessing.get_sharing_strategy())
    main()
