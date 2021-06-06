#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model
author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataflow.consep import ConSep


from setting import ConSepSettings
#from common.utils import get_network, get_test_dataloader
def get_network(network_name, num_classes, pretrained):
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
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=10, help='batch size for dataloader')
    args = parser.parse_args()

    settings = ConSepSettings()


    valid_dataset = ConSep(
        settings.test_root
    )
    print()
    mean = valid_dataset.mean
    std = valid_dataset.std

    valid_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((settings.image_size, settings.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    valid_dataset.transforms = valid_transforms


    validation_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.b, num_workers=4)

    net = get_network(args.net, 4, False)
    net.load_state_dict(torch.load(args.weights))
    if args.gpu:
        net = net.cuda()
    #cifar100_test_loader = get_test_dataloader(
    #    settings.CIFAR100_TRAIN_MEAN,
    #    settings.CIFAR100_TRAIN_STD,
    #    #settings.CIFAR100_PATH,
    #    num_workers=4,
    #    batch_size=args.b,
    #)

    print(net)
    net.eval()

    correct = 0.0
    #correct_5 = 0.0
    #total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(validation_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(validation_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                #print('GPU INFO.....')
                #print(torch.cuda.memory_summary(), end='')


            outputs = net(image)

            _, preds = outputs.max(1)
            correct += preds.eq(label).sum()
            #print(output.shape)
            #_, pred = output.topk(5, 1, largest=True, sorted=True)

            #label = label.view(label.size(0), -1).expand_as(pred)
            #correct = pred.eq(label).float()

            ##compute top 5
            #correct_5 += correct[:, :5].sum()

            ##compute top1
            #correct_1 += correct[:, :1].sum()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 err: ", 1 - correct / len(validation_loader.dataset))
    #print("Top 5 err: ", 1 - correct_5 / len(validation_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))