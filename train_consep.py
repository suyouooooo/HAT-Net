import argparse
import os
import time
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from torchvision import transforms
from setting import ConSepSettings
#from dataset.camvid import CamVid
#from dataset.camvid_lmdb import CamVid
from dataflow.consep import ConSep
import common.utils as utils
from common.lr import WarmUpLR



def network(network_name, num_classes, pretrained):
    if network_name == 'resnet34':
        from model.resnet import resnet34
        net = resnet34(pretrained=pretrained)
        if num_classes != 1000:
            net.reset_num_classes(num_classes)
    elif network_name == 'resnet50':
        from model.resnet import resnet50
        net = resnet50(pretrained=pretrained)
        if num_classes != 1000:
            net.reset_num_classes(num_classes)
    else:
        raise ValueError('network names not suppored')
    return net

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=10,
                        help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=5e-4,
                        help='initial learning rate')
    parser.add_argument('-e', type=int, default=120, help='training epoches')
    parser.add_argument('-net', type=str, required=True, help='if resume training')
    parser.add_argument('-warm', type=int, default=5, help='')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')

    args = parser.parse_args()
    settings = ConSepSettings()

    root_path = os.path.dirname(os.path.abspath(__file__))

    checkpoint_path = os.path.join(
        root_path, settings.checkpoint, args.net, settings.timenow)
    log_dir = os.path.join(root_path, settings.log_path, args.net, settings.timenow)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{epoch}-{type}.pth')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    train_dataset = ConSep(
        settings.train_root
    )
    valid_dataset = ConSep(
        settings.test_root
    )
    print()
    mean = train_dataset.mean
    std = train_dataset.std

    train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.RandomResizedCrop(settings.image_size),
            transforms.RandomApply(torch.nn.ModuleList([
                    transforms.ColorJitter()
            ]), p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)

    ])

    valid_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((settings.image_size, settings.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset.transforms = train_transforms
    valid_dataset.transforms = valid_transforms

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.b, num_workers=4, shuffle=True)

    validation_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.b, num_workers=4)

    net = network(args.net, train_dataset.num_classes, pretrained=True)

    #if args.resume:
    #    weight_path = utils.get_weight_path(
    #        os.path.join(root_path, settings.CHECKPOINT_FOLDER))
    #    print('Loading weight file: {}...'.format(weight_path))
    #    net.load_state_dict(torch.load(weight_path))
    #    print('Done loading!')

    if args.gpu:
        net = net.cuda()

    tensor = torch.Tensor(1, 3, settings.image_size, settings.image_size)
    utils.visualize_network(writer, net, tensor)

    #optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    iter_per_epoch = len(train_loader)

    #loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.milestones, gamma=0.2) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)


    #train_scheduler = optim.lr_scheduler.OneCycleLR(
    #    optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.e)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0

    trained_epochs = 0

    #if args.resume:
    #    trained_epochs = int(
    #        re.search('([0-9]+)-(best|regular).pth', weight_path).group(1))
    #    train_scheduler.step(trained_epochs * len(train_loader))

    for epoch in range(trained_epochs + 1, args.e + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        start = time.time()

        net.train()

        ious = 0
        for batch_idx, (images, masks) in enumerate(train_loader):

            optimizer.zero_grad()

            if args.gpu:
                images = images.cuda()
                masks = masks.cuda()
            preds = net(images)

            loss = loss_fn(preds, masks)
            loss.backward()


           # print(('Training Epoch:{epoch} [{trained_samples}/{total_samples}] '
           #         'Lr:{lr:0.6f} Loss:{loss:0.4f}).format(
           #     loss=loss.item(),
           #     epoch=epoch,
           #     trained_samples=batch_idx * args.b + len(images),
           #     total_samples=len(train_dataset),
           #     lr=optimizer.param_groups[0]['lr'],
           #     #beta=optimizer.param_groups[0]['betas'][0]
           # ))
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_idx * args.b + len(images),
                total_samples=len(train_loader.dataset)
            ))


            n_iter = (epoch - 1) * iter_per_epoch + batch_idx + 1
            utils.visualize_lastlayer(
                writer,
                net,
                n_iter,
            )

            optimizer.step()
            if epoch <= args.warm:
                warmup_scheduler.step()

        utils.visualize_scalar(
            writer,
            'Train/LearningRate',
            optimizer.param_groups[0]['lr'],
            epoch,
        )

        #utils.visualize_scalar(
        #    writer,
        #    'Train/Beta1',
        #    optimizer.param_groups[0]['betas'][0],
        #    epoch,
        #)
        utils.visualize_param_hist(writer, net, epoch)
        print('time for training epoch {} : {:.2f}s'.format(epoch, time.time() - start))

        net.eval()
        test_loss = 0.0

        test_start = time.time()
        acc = 0
        correct = 0.0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(validation_loader):

                if args.gpu:
                    images = images.cuda()
                    labels = labels.cuda()

                outputs = net(images)

                loss = loss_fn(outputs, labels)
                test_loss += loss.item()

                test_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum()

        test_finish = time.time()
        acc = correct.float() / len(validation_loader.dataset)
        if args.gpu:
            print('GPU INFO.....')
            print(torch.cuda.memory_summary(), end='')
        print('Evaluating Network.....')
        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss / len(validation_loader.dataset),
            acc,
            test_finish - test_start
        ))
        print()


        utils.visualize_scalar(
            writer,
            'Test/Acc',
            acc,
            epoch,
        )

        utils.visualize_scalar(
            writer,
            'Test/Loss',
            test_loss / len(valid_dataset),
            epoch,
        )

        if best_acc < acc and epoch > args.e // 20:
            best_acc = acc
            torch.save(net.state_dict(),
                            checkpoint_path.format(epoch=epoch, type='best'))
            print('saving file to {}'.format(checkpoint_path.format(epoch=epoch, type='best')))
            continue

        if not epoch % settings.save_epoch:
            torch.save(net.state_dict(),
                            checkpoint_path.format(epoch=epoch, type='regular'))