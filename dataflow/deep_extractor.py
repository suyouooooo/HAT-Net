import torch
from torch.nn.functional import adaptive_avg_pool3d, adaptive_avg_pool2d, adaptive_avg_pool1d


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


#class ExtractorResNet50:
#    def __init__(self, weight_path, num_classes, output_dim=None):
#        #self.net = network('resnet50',  5, False)
#        self.net = network('resnet50', num_classes, False)
#        self.net = self.net.cuda()
#        self.net = torch.nn.DataParallel(self.net)
#        print('loading weight file {}...'.format(weight_path))
#        self.net.load_state_dict(torch.load(weight_path))
#        print('Done.')
#        self.output_dim = output_dim
#
#    def __call__(self, images):
#        with torch.no_grad():
#            output = self.net(images.cuda())
#            output = output.unsqueeze(0)
#            if self.output_dim is not None:
#                output = adaptive_avg_pool3d(output, (self.output_dim, 1, 1))
#            output = output.squeeze()
#
#        return output.cpu().numpy()
        #return output.cpu()

class ExtractorResNet50ImageNet:
    def __init__(self, num_classes, output_dim=None):
        self.net = network('resnet50',  5, True)
        self.net = self.net.cuda()
        print(torch.cuda.device_count())
        self.net = torch.nn.DataParallel(self.net)

    def __call__(self, images):
        with torch.no_grad():
            output = self.net(images.cuda())
            output = output.unsqueeze(0)
            if self.output_dim is not None:
                output = adaptive_avg_pool3d(output, (self.output_dim, 1, 1))
            output = output.squeeze()

        return output.cpu().numpy()

class ExtractorResNet50:
    def __init__(self, weight_path, num_classes, output_dim=None):
        #self.net = network('resnet50',  5, False)
        self.net = network('resnet50', num_classes, False)
        self.net = self.net.cuda()
        self.net = torch.nn.DataParallel(self.net)
        print('loading weight file {}...'.format(weight_path))
        self.net.load_state_dict(torch.load(weight_path))
        print('Done.')
        self.output_dim = output_dim

    def __call__(self, images):
        with torch.no_grad():
            output = self.net(images.cuda())
            B, C, _, _ = output.shape
            output = output.reshape(B, C)
            if self.output_dim is not None:
                output = output.reshape(1, B, C, 1, 1)
                output = adaptive_avg_pool3d(output, (self.output_dim, 1, 1))
                output = output.reshape(B, self.output_dim)

        return output.cpu().numpy()


class ExtractorVGG:
    def __init__(self, weight_path, num_classes, output_dim=None):
        import segmentation_models_pytorch as smp
        import torch.nn as nn
        from torch.nn.functional import adaptive_avg_pool2d

        self.relu_type = nn.ReLU

        model = smp.Unet(
            encoder_name="vgg16_bn",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,                      # model output channels (number of classes in your dataset)
        )
        self.pool = adaptive_avg_pool2d
        #weight_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/checkpoints/vggunet/183-best.pth'
        print('loading file {}.....'.format(weight_path))
        model.load_state_dict(torch.load(weight_path))
        print('done')
        self.encoder = model.encoder.features[:13]
        self.encoder = self.encoder.cuda()
        self.encoder = torch.nn.DataParallel(self.encoder)
        self.pool = adaptive_avg_pool2d
        self.output_dim = output_dim


    def extract_feature(self, output):
        res = []
        assert output.size(2) == 71
        assert output.size(3) == 71
        with torch.no_grad():
            for layer in self.encoder.module:
                output = layer(output)
                if type(layer) == self.relu_type:
                    tmp = output
                    B, C, _, _ = tmp.shape
                    tmp = self.pool(tmp, (1, 1))
                    tmp = tmp.reshape(B, C)
                    res.append(tmp)

        res = torch.cat(res, dim=1)
        return res


    def __call__(self, images):
        output = self.extract_feature(images.cuda())

        if self.output_dim is not None:
            output = output.unsqueeze(0)
            output = adaptive_avg_pool1d(output, (self.output_dim))
            output = output.squeeze(0)

        return output.cpu().numpy()
