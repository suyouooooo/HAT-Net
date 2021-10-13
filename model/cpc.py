from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

#from GreedyInfoMax.vision.models import PixelCNN_Autoregressor, Resnet_Encoder
from torch.nn.modules.loss import _WeightedLoss
import torchvision.transforms as transforms




#from GreedyInfoMax.vision.models import InfoNCE_Loss, Supervised_Loss
#from GreedyInfoMax.utils import model_utils

def get_transforms():
    aug = {
        "stl10": {
            "randcrop": 64,
            #"randcrop": 256,
            #"flip": True,
            #"flip": True,
            "grayscale": True,
            "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined
            "std": [0.2683, 0.2610, 0.2687],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
    }
    trans = []

    aug = aug['stl10']
    trans.append(transforms.ToPILImage())
    #trans.append(transforms.Resize((aug["randcrop"])))
    trans.append(transforms.CenterCrop(64))
    #trans.append(transforms.Resize((aug["randcrop"])))
    #trans.append(transforms.RandomCrop(aug["randcrop"]))
    #if aug["randcrop"] and not eval:
    #    trans.append(transforms.RandomCrop(aug["randcrop"]))

    #if aug["randcrop"] and eval:
    #    trans.append(transforms.CenterCrop(aug["randcrop"]))

    #if aug["flip"] and not eval:
    #    trans.append(transforms.RandomHorizontalFlip())

    #print(aug.keys())
    if aug["grayscale"]:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
    elif aug["mean"]:
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
    else:
        trans.append(transforms.ToTensor())

    trans = transforms.Compose(trans)
    return trans


class ResNet_Encoder(nn.Module):
    def __init__(
        self,
        opt,
        block,
        num_blocks,
        filter,
        encoder_num,
        #patch_size=16,
        patch_size=64,
        input_dims=3,
        calc_loss=False,
    ):
        super(ResNet_Encoder, self).__init__()
        self.encoder_num = encoder_num
        self.opt = opt

        self.patchify = True
        self.overlap = 2

        self.calc_loss = calc_loss
        self.patch_size = patch_size
        self.filter = filter

        self.model = nn.Sequential()

        if encoder_num == 0:
            self.model.add_module(
                "Conv1",
                nn.Conv2d(
                    input_dims, self.filter[0], kernel_size=5, stride=1, padding=2
                ),
            )
            self.in_planes = self.filter[0]
            self.first_stride = 1
        elif encoder_num > 2:
            self.in_planes = self.filter[0] * block.expansion
            self.first_stride = 2
        else:
            self.in_planes = (self.filter[0] // 2) * block.expansion
            self.first_stride = 2

        for idx in range(len(num_blocks)):
            self.model.add_module(
                "layer {}".format((idx)),
                self._make_layer(
                    block, self.filter[idx], num_blocks[idx], stride=self.first_stride
                ),
            )
            self.first_stride = 2

        ## loss module is always present, but only gets used when training GreedyInfoMax modules
        if self.opt.loss == 0:
            self.loss = InfoNCE_Loss(
                opt,
                in_channels=self.in_planes,
                out_channels=self.in_planes
            )
        elif self.opt.loss == 1:
            self.loss = Supervised_Loss.Supervised_Loss(opt, self.in_planes, True)
        else:
            raise Exception("Invalid option")

        if self.opt.weight_init:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                model_utils.makeDeltaOrthogonal(
                    m.weight, nn.init.calculate_gain("relu")
                )
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                m.momentum = 0.3

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x, n_patches_x, n_patches_y, label, patchify_right_now=True):
        if self.patchify and self.encoder_num == 0 and patchify_right_now:
            #print('here', self.patch_size, self.patch_size // self.overlap)
            #x = [:, :, :64, :64]
            #print(x.shape)
            #self.patch_size = 16
            #self.overlap = 2
            x = (
                x.unfold(2, self.patch_size, self.patch_size // self.overlap)
                .unfold(3, self.patch_size, self.patch_size // self.overlap)
                .permute(0, 2, 3, 1, 4, 5)
            )
            n_patches_x = x.shape[1]
            n_patches_y = x.shape[2]
            x = x.reshape(
                x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
            )

        #print(x.shape)
        z = self.model(x)

        out = F.adaptive_avg_pool2d(z, 1)
        out = out.reshape(-1, n_patches_x, n_patches_y, out.shape[1])
        out = out.permute(0, 3, 1, 2).contiguous()

        accuracy = torch.zeros(1)
        if self.calc_loss and self.opt.loss == 0:
            loss = self.loss(out, out)
        elif self.calc_loss and self.opt.loss == 1:
            loss, accuracy = self.loss(out, label)
        else:
            loss = None

        return out, z, loss, accuracy, n_patches_x, n_patches_y



def genOrthgonal(dim):
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q


def makeDeltaOrthogonal(weights, gain):
    rows = weights.size(0)
    cols = weights.size(1)
    if rows > cols:
        print("In_filters should not be greater than out_filters.")
    weights.data.fill_(0)
    dim = max(rows, cols)
    q = genOrthgonal(dim)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    with torch.no_grad():
        weights[:, :, mid1, mid2] = q[: weights.size(0), : weights.size(1)]
        weights.mul_(gain)


def reload_weights(opt, model, optimizer, reload_model):
    ## reload weights for training of the linear classifier
    if (opt.model_type == 0) and reload_model:  # or opt.model_type == 2)
        print("Loading weights from ", opt.model_path)

        if opt.experiment == "audio":
            model.load_state_dict(
                torch.load(
                    os.path.join(opt.model_path, "model_{}.ckpt".format(opt.model_num)),
                    map_location=opt.device.type,
                )
            )
        else:
            for idx, layer in enumerate(model.module.encoder):
                model.module.encoder[idx].load_state_dict(
                    torch.load(
                        os.path.join(
                            opt.model_path,
                            "model_{}_{}.ckpt".format(idx, opt.model_num),
                        ),
                         map_location=opt.device.type,
                    )
                )

    ## reload weights and optimizers for continuing training
    elif opt.start_epoch > 0:
        print("Continuing training from epoch ", opt.start_epoch)

        if opt.experiment == "audio":
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        opt.model_path, "model_{}.ckpt".format(opt.start_epoch)
                    ),
                    map_location=opt.device.type,
                ),
                strict=False,
            )
        else:
            for idx, layer in enumerate(model.module.encoder):
                model.module.encoder[idx].load_state_dict(
                    torch.load(
                        os.path.join(
                            opt.model_path,
                            "model_{}_{}.ckpt".format(idx, opt.start_epoch),
                        ),
                        map_location=opt.device.type,
                    )
                )

        optimizer.load_state_dict(
            torch.load(
                os.path.join(
                    opt.model_path,
                    "optim_{}.ckpt".format(opt.start_epoch),
                ),
                map_location=opt.device.type,
            )
        )
    else:
        print("Randomly initialized model")

    return model, optimizer


class InfoNCE_Loss(nn.Module):
    def __init__(self, opt, in_channels, out_channels):
        super().__init__()
        self.opt = opt
        self.negative_samples = self.opt.negative_samples
        self.k_predictions = self.opt.prediction_step

        self.W_k = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range(self.k_predictions)
        )

        self.contrast_loss = ExpNLLLoss()

        if self.opt.weight_init:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                if m in self.W_k:
                    # nn.init.kaiming_normal_(
                    #     m.weight, mode="fan_in", nonlinearity="tanh"
                    # )
                    model_utils.makeDeltaOrthogonal(
                        m.weight,
                        nn.init.calculate_gain(
                            "Sigmoid"
                        ),
                    )

    def forward(self, z, c, skip_step=1):

        batch_size = z.shape[0]

        total_loss = 0

        if self.opt.device.type != "cpu":
            cur_device = z.get_device()
        else:
            cur_device = self.opt.device

        # For each element in c, contrast with elements below
        for k in range(1, self.k_predictions + 1):
            ### compute log f(c_t, x_{t+k}) = z^T_{t+k} W_k c_t
            # compute z^T_{t+k} W_k:
            ztwk = (
                self.W_k[k - 1]
                .forward(z[:, :, (k + skip_step) :, :])  # Bx, C , H , W
                .permute(2, 3, 0, 1)  # H, W, Bx, C
                .contiguous()
            )  # y, x, b, c

            ztwk_shuf = ztwk.view(
                ztwk.shape[0] * ztwk.shape[1] * ztwk.shape[2], ztwk.shape[3]
            )  # y * x * batch, c
            rand_index = torch.randint(
                ztwk_shuf.shape[0],  # y *  x * batch
                (ztwk_shuf.shape[0] * self.negative_samples, 1),
                dtype=torch.long,
                device=cur_device,
            )
            # Sample more
            rand_index = rand_index.repeat(1, ztwk_shuf.shape[1])

            ztwk_shuf = torch.gather(
                ztwk_shuf, dim=0, index=rand_index, out=None
            )  # y * x * b * n, c

            ztwk_shuf = ztwk_shuf.view(
                ztwk.shape[0],
                ztwk.shape[1],
                ztwk.shape[2],
                self.negative_samples,
                ztwk.shape[3],
            ).permute(
                0, 1, 2, 4, 3
            )  # y, x, b, c, n

            #### Compute  x_W1 . c_t:
            context = (
                c[:, :, : -(k + skip_step), :].permute(2, 3, 0, 1).unsqueeze(-2)
            )  # y, x, b, 1, c

            log_fk_main = torch.matmul(context, ztwk.unsqueeze(-1)).squeeze(
                -2
            )  # y, x, b, 1

            log_fk_shuf = torch.matmul(context, ztwk_shuf).squeeze(-2)  # y, x, b, n

            log_fk = torch.cat((log_fk_main, log_fk_shuf), 3)  # y, x, b, 1+n
            log_fk = log_fk.permute(2, 3, 0, 1)  # b, 1+n, y, x

            log_fk = torch.softmax(log_fk, dim=1)

            true_f = torch.zeros(
                (batch_size, log_fk.shape[-2], log_fk.shape[-1]),
                dtype=torch.long,
                device=cur_device,
            )  # b, y, x

            total_loss += self.contrast_loss(input=log_fk, target=true_f)

        total_loss /= self.k_predictions

        return total_loss


class ExpNLLLoss(_WeightedLoss):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(ExpNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        x = torch.log(input + 1e-11)
        return F.nll_loss(x, target, weight=self.weight, ignore_index=self.ignore_index,
                          reduction=self.reduction)


class PreActBlockNoBN(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlockNoBN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride
                )
            )

    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += shortcut
        return out


class PreActBottleneckNoBN(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneckNoBN, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        # self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride
                )
            )

    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(out))
        out = self.conv3(F.relu(out))
        out += shortcut
        return out


class ResNet_Encoder(nn.Module):
    def __init__(
        self,
        opt,
        block,
        num_blocks,
        filter,
        encoder_num,
        #patch_size=16,
        patch_size=64,
        input_dims=3,
        calc_loss=False,
    ):
        super(ResNet_Encoder, self).__init__()
        self.encoder_num = encoder_num
        self.opt = opt

        self.patchify = True
        self.overlap = 2

        self.calc_loss = calc_loss
        self.patch_size = patch_size
        self.filter = filter

        self.model = nn.Sequential()

        if encoder_num == 0:
            self.model.add_module(
                "Conv1",
                nn.Conv2d(
                    input_dims, self.filter[0], kernel_size=5, stride=1, padding=2
                ),
            )
            self.in_planes = self.filter[0]
            self.first_stride = 1
        elif encoder_num > 2:
            self.in_planes = self.filter[0] * block.expansion
            self.first_stride = 2
        else:
            self.in_planes = (self.filter[0] // 2) * block.expansion
            self.first_stride = 2

        for idx in range(len(num_blocks)):
            self.model.add_module(
                "layer {}".format((idx)),
                self._make_layer(
                    block, self.filter[idx], num_blocks[idx], stride=self.first_stride
                ),
            )
            self.first_stride = 2

        ## loss module is always present, but only gets used when training GreedyInfoMax modules
        if self.opt.loss == 0:
            self.loss = InfoNCE_Loss(
                opt,
                in_channels=self.in_planes,
                out_channels=self.in_planes
            )
        elif self.opt.loss == 1:
            self.loss = Supervised_Loss.Supervised_Loss(opt, self.in_planes, True)
        else:
            raise Exception("Invalid option")

        if self.opt.weight_init:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                model_utils.makeDeltaOrthogonal(
                    m.weight, nn.init.calculate_gain("relu")
                )
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                m.momentum = 0.3

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x, n_patches_x, n_patches_y, label, patchify_right_now=True):
        if self.patchify and self.encoder_num == 0 and patchify_right_now:
            #print('here', self.patch_size, self.patch_size // self.overlap)
            #x = [:, :, :64, :64]
            #print(x.shape)
            #self.patch_size = 16
            #self.overlap = 2
            x = (
                x.unfold(2, self.patch_size, self.patch_size // self.overlap)
                .unfold(3, self.patch_size, self.patch_size // self.overlap)
                .permute(0, 2, 3, 1, 4, 5)
            )
            n_patches_x = x.shape[1]
            n_patches_y = x.shape[2]
            x = x.reshape(
                x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
            )

        #print(x.shape)
        z = self.model(x)

        out = F.adaptive_avg_pool2d(z, 1)
        out = out.reshape(-1, n_patches_x, n_patches_y, out.shape[1])
        out = out.permute(0, 3, 1, 2).contiguous()

        accuracy = torch.zeros(1)
        if self.calc_loss and self.opt.loss == 0:
            loss = self.loss(out, out)
        elif self.calc_loss and self.opt.loss == 1:
            loss, accuracy = self.loss(out, label)
        else:
            loss = None

        return out, z, loss, accuracy, n_patches_x, n_patches_y

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#owidth  = floor((width  + 2*padW - kW) / dW + 1)
#oheight = floor((height + 2*padH - kH) / dH + 1)
#dW is stride, assuming 1:
# kW // 2 = padW
def same_padding(kernel_size):
    # assumming stride 1
    if isinstance(kernel_size, int):
        return kernel_size // 2
    else:
        return (kernel_size[0] // 2, kernel_size[1] // 2)

# PyTorch port of
class MaskedConvolution2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
            *args, mask='B', vertical=False, mask_mode="noblind", **kwargs):
        if "padding" not in kwargs:
            assert "stride" not in kwargs
            kwargs["padding"] = same_padding(kernel_size)
        remove = {"conditional_features", "conditional_image_channels"}
        for feature in remove:
            if feature in kwargs:
                del kwargs[feature]
        super(MaskedConvolution2D, self).__init__(in_channels,
                out_channels, kernel_size, *args, **kwargs)
        Cout, Cin, kh, kw = self.weight.size()
        pre_mask = np.ones_like(self.weight.data.cpu().numpy()).astype(np.float32)
        yc, xc = kh // 2, kw // 2

        assert mask_mode in {"noblind", "turukin", "fig1-van-den-oord", "none", "only_vert"}

        if mask_mode == "none":
            pass
        elif mask_mode == "only_vert":
            pre_mask[:, :, yc + 1:, :] = 0.0
        elif mask_mode == "noblind":
            # context masking - subsequent pixels won't have access
            # to next pixels (spatial dim)
            if vertical:
                if mask == 'A':
                    # In the first layer, can ONLY access pixels above it
                    pre_mask[:, :, yc:, :] = 0.0
                else:
                    # In the second layer, can access pixels above or even with it.
                    # Reason being that the pixels to the right or left of the current pixel
                    #  only have a receptive field of the layer above the current layer and up.
                    pre_mask[:, :, yc+1:, :] = 0.0
            else:
                # All rows after center must be zero
                pre_mask[:, :, yc+1:, :] = 0.0
                ### All rows before center must be zero # XXX: not actually necessary
                ##pre_mask[:, :, :yc, :] = 0.0
                # All columns after center in center row must be zero
                pre_mask[:, :, yc, xc+1:] = 0.0

            if mask == 'A':
                # Center must be zero in first layer
                pre_mask[:, :, yc, xc] = 0.0
            # same pixel masking - pixel won't access next color (conv filter dim)
            #def bmask(i_out, i_in):
            #    cout_idx = np.expand_dims(np.arange(Cout) % 3 == i_out, 1)
            #    cin_idx = np.expand_dims(np.arange(Cin) % 3 == i_in, 0)
            #    a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
            #    return a1 * a2

            #for j in range(3):
            #    pre_mask[bmask(j, j), yc, xc] = 0.0 if mask == 'A' else 1.0

            #pre_mask[bmask(0, 1), yc, xc] = 0.0
            #pre_mask[bmask(0, 2), yc, xc] = 0.0
            #pre_mask[bmask(1, 2), yc, xc] = 0.0
        elif mask_mode == "fig1-van-den-oord":
            if vertical:
                pre_mask[:, :, yc:, :] = 0.0
            else:
                # All rows after center must be zero
                pre_mask[:, :, yc+1:, :] = 0.0
                ### All rows before center must be zero # XXX: not actually necessary
                ##pre_mask[:, :, :yc, :] = 0.0
                # All columns after center in center row must be zero
                pre_mask[:, :, yc, xc+1:] = 0.0

            if mask == 'A':
                # Center must be zero in first layer
                pre_mask[:, :, yc, xc] = 0.0
        elif mask_mode == "turukin":
            pre_mask[:, :, yc+1:, :] = 0.0
            pre_mask[:, :, yc, xc+1:] = 0.0
            if mask == 'A':
                pre_mask[:, :, yc, xc] = 0.0

        print("%s %s MASKED CONV: %d x %d. Mask:" % (mask, "VERTICAL" if vertical else "HORIZONTAL", kh, kw))
        print(pre_mask[0, 0, :, :])

        self.register_buffer("mask", torch.from_numpy(pre_mask))

    def __call__(self, x):
        self.weight.data = self.weight.data * self.mask
        return super(MaskedConvolution2D, self).forward(x)


class PixelCNNGatedLayer(nn.Module):
    def __init__(self, primary, in_channels, out_channels, filter_size,
            mask='B', nobias=False, conditional_features=None,
            conditional_image_channels=None, residual_vertical=False,
            residual_horizontal=True, skips=False, gated=True,
            relu_out=False, horizontal_2d_convs=False, mask_mode="noblind", groups=1):
        super().__init__()
        self.primary = primary
        if primary:
            assert mask == 'A'
            assert not residual_vertical
            assert not residual_horizontal
        else:
            assert mask == 'B'
        self.out_channels = out_channels
        self.gated = gated
        gm = 2 if gated else 1
        self.vertical_conv = MaskedConvolution2D(
            in_channels, gm * out_channels, (filter_size, filter_size),
            mask=mask, vertical=True, mask_mode=mask_mode, groups=groups)
        self.v_to_h_conv = nn.Conv2d(gm * out_channels, gm * out_channels, 1, groups=groups)

        self.horizontal_conv = MaskedConvolution2D(
            in_channels, gm * out_channels,
            (filter_size if horizontal_2d_convs else 1, filter_size), # XXX: traditionally (1, filter_size),
            mask=mask, vertical=False, mask_mode=mask_mode, groups=groups)

        self.residual_vertical = None
        if residual_vertical:
            self.residual_vertical = nn.Conv2d(in_channels, gm * out_channels, 1, groups=groups)

        self.horizontal_output = nn.Conv2d(out_channels, out_channels, 1, groups=groups)
        self.horizontal_skip = None
        if skips:
            self.horizontal_skip = nn.Conv2d(out_channels, out_channels, 1, groups=groups)
        self.conditional_vector = conditional_features is not None
        self.conditional_image = conditional_image_channels is not None
        if self.conditional_image:
            self.cond_conv_h = nn.Conv2d(conditional_image_channels, gm * out_channels, 1, bias=False, groups=groups)
            self.cond_conv_v = nn.Conv2d(conditional_image_channels, gm * out_channels, 1, bias=False, groups=groups)
        if self.conditional_vector:
            self.cond_fc_h = nn.Linear(conditional_features, gm * out_channels, bias=False)
            self.cond_fc_v = nn.Linear(conditional_features, gm * out_channels, bias=False)
        self.residual_horizontal = residual_horizontal
        self.relu_out = relu_out

    @classmethod
    def primary(cls, in_channels, out_channels, filter_size,
            nobias=False, conditional_features=None,
            conditional_image_channels=None,
            skips=False, gated=True,
            relu_out=False, horizontal_2d_convs=False, mask_mode="noblind", groups=1):
        return cls(True, in_channels, out_channels, filter_size, nobias=nobias,
                mask='A', conditional_features=conditional_features,
                conditional_image_channels=conditional_image_channels,
                residual_vertical=False, residual_horizontal=False,
                skips=skips, gated=gated,
                relu_out=relu_out, horizontal_2d_convs=horizontal_2d_convs,
                mask_mode=mask_mode, groups=groups)

    @classmethod
    def secondary(cls, in_channels, out_channels, filter_size,
            nobias=False, conditional_features=None,
            conditional_image_channels=None, residual_vertical=True,
            residual_horizontal=True, skips=False, gated=True,
            relu_out=False, horizontal_2d_convs=False, mask_mode="noblind", groups=1):
        return cls(False, in_channels, out_channels, filter_size, nobias=nobias,
                mask='B', conditional_features=conditional_features,
                conditional_image_channels=conditional_image_channels,
                residual_vertical=residual_vertical, residual_horizontal=residual_horizontal,
                skips=skips, gated=gated, relu_out=relu_out,
                horizontal_2d_convs=horizontal_2d_convs, mask_mode=mask_mode, groups=groups)

    def _gate(self, x):
        if self.gated:
            return torch.tanh(x[:,:self.out_channels]) * torch.sigmoid(x[:,self.out_channels:])
        else:
            return x

    def __call__(self, v, h, conditional_image=None, conditional_vector=None):
        horizontal_preactivation = self.horizontal_conv(h) # 1xN
        vertical_preactivation = self.vertical_conv(v) # NxN
        v_to_h = self.v_to_h_conv(vertical_preactivation) # 1x1
        if self.residual_vertical is not None:
            vertical_preactivation = vertical_preactivation + self.residual_vertical(v) # 1x1 to residual
        horizontal_preactivation = horizontal_preactivation + v_to_h
        if self.conditional_image and conditional_image is not None:
            horizontal_preactivation = horizontal_preactivation + \
                    self.cond_conv_h(conditional_image)
            vertical_preactivation = vertical_preactivation + \
                    self.cond_conv_v(conditional_image)
        if self.conditional_vector and conditional_vector is not None:
            horizontal_preactivation = horizontal_preactivation + \
                    self.cond_fc_h(conditional_vector).unsqueeze(-1).unsqueeze(-1)
            vertical_preactivation = vertical_preactivation + \
                    self.cond_fc_v(conditional_vector).unsqueeze(-1).unsqueeze(-1)
        v_out = self._gate(vertical_preactivation)
        h_activated = self._gate(horizontal_preactivation)
        h_skip = None
        if self.horizontal_skip is not None:
            h_skip = self.horizontal_skip(h_activated)
        h_preres = self.horizontal_output(h_activated)
        if self.residual_horizontal:
            h_out = h + h_preres
        else:
            h_out = h_preres
        if self.relu_out:
            v_out = F.relu(v_out)
            h_out = F.relu(h_out)
            if h_skip is not None:
                h_skip = F.relu(h_skip)
        return v_out, h_out, h_skip

class PixelCNNGatedStack(nn.Module):
    def __init__(self, *args):
        super().__init__()
        layers = list(args)
        for i, layer in enumerate(layers):
            assert isinstance(layer, PixelCNNGatedLayer)
            if i == 0:
                assert layer.primary
            else:
                assert not layer.primary
        self.layers = nn.ModuleList(layers)

    def __call__(self, v, h, skips=None, conditional_image=None, conditional_vector=None):
        if skips is None:
            skips = []
        else:
            skips = [skips]
        for layer in self.layers:
            v, h, skip = layer(v, h, conditional_image=conditional_image, conditional_vector=conditional_vector)
            if skip is not None:
                skips.append(skip)
        if len(skips) == 0:
            skips = None
        else:
            skips = torch.cat(skips, 1)
        return v, h, skips



class PixelCNN_Autoregressor(torch.nn.Module):
    def __init__(self, opt, in_channels, pixelcnn_layers=4, calc_loss=True, **kwargs):
        super().__init__()
        self.opt = opt
        self.calc_loss = calc_loss

        layer_objs = [
            PixelCNNGatedLayer.primary(
                in_channels, in_channels, 3, mask_mode="only_vert", **kwargs
            )
        ]
        layer_objs = layer_objs + [
            PixelCNNGatedLayer.secondary(
                in_channels, in_channels, 3, mask_mode="only_vert", **kwargs
            )
            for _ in range(1, pixelcnn_layers)
        ]

        self.stack = PixelCNNGatedStack(*layer_objs)
        self.stack_out = nn.Conv2d(in_channels, in_channels, 1)

        self.loss = InfoNCE_Loss(
            opt, in_channels=in_channels, out_channels=in_channels
        )

        if self.opt.weight_init:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                if m is self.stack_out:
                    # nn.init.kaiming_normal_(
                    #     m.weight, mode="fan_in", nonlinearity="relu"
                    # )
                    model_utils.makeDeltaOrthogonal(
                        m.weight, nn.init.calculate_gain("relu")
                    )
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="tanh"
                    )
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                m.momentum = 0.3

    def forward(self, input):
        _, c_out, _ = self.stack(input, input)  # Bc, C, H, W
        c_out = self.stack_out(c_out)

        assert c_out.shape[1] == input.shape[1]

        if self.calc_loss:
            loss = self.loss(input, c_out)
        else:
            loss = None

        return c_out, loss


class FullVisionModel(torch.nn.Module):
    def __init__(self, opt, calc_loss):
        super().__init__()
        self.opt = opt
        self.contrastive_samples = self.opt.negative_samples
        print("Contrasting against ", self.contrastive_samples, " negative samples")
        self.calc_loss = calc_loss

        if self.opt.model_splits == 1 and not self.opt.loss == 1:
            # building the CPC model including the autoregressive PixelCNN on top of the ResNet
            self.employ_autoregressive = True
        else:
            self.employ_autoregressive = False

        self.model, self.encoder, self.autoregressor = self._create_full_model(opt)

        print(self.model)

    def _create_full_model(self, opt):

        block_dims = [3, 4, 6]
        num_channels = [64, 128, 256]

        full_model = nn.ModuleList([])
        encoder = nn.ModuleList([])

        if opt.resnet == 34:
            self.block = PreActBlockNoBN
        elif opt.resnet == 50:
            self.block = PreActBottleneckNoBN
        else:
            raise Exception("Undefined parameter choice")

        if opt.grayscale:
            input_dims = 1
        else:
            input_dims = 3

        output_dims = num_channels[-1] * self.block.expansion

        if opt.model_splits == 1:
            encoder.append(
                ResNet_Encoder(
                    opt,
                    self.block,
                    block_dims,
                    num_channels,
                    0,
                    calc_loss=False,
                    input_dims=input_dims,
                )
            )
        elif opt.model_splits == 3:
            for idx, _ in enumerate(block_dims):
                encoder.append(
                    Resnet_Encoder.ResNet_Encoder(
                        opt,
                        self.block,
                        [block_dims[idx]],
                        [num_channels[idx]],
                        idx,
                        calc_loss=False,
                        input_dims=input_dims,
                    )
                )
        else:
            raise NotImplementedError

        full_model.append(encoder)

        if self.employ_autoregressive:
            autoregressor = PixelCNN_Autoregressor(
                opt, in_channels=output_dims, calc_loss=True
            )

            full_model.append(autoregressor)
        else:
            autoregressor = None

        return full_model, encoder, autoregressor


    def forward(self, x, label=10, n=3):
        model_input = x

        if self.opt.device.type != "cpu":
            cur_device = x.get_device()
        else:
            cur_device = self.opt.device

        n_patches_x, n_patches_y = None, None

        #print(cur_device)

        for idx, module in enumerate(self.encoder[: n+1]):
            h, z, cur_loss, cur_accuracy, n_patches_x, n_patches_y = module(
                model_input, n_patches_x, n_patches_y, label
            )
            # Detach z to make sure no gradients are flowing in between modules
            # we can detach z here, as for the CPC model the loop is only called once and h is forward-propagated
            model_input = z.detach()

            #if cur_loss is not None:
            #    loss[:, idx] = cur_loss
            #    accuracies[:, idx] = cur_accuracy

        return h
        #print(self.employ_autoregressive, self.calc_loss, 11111)
        loss = torch.zeros(1, self.opt.model_splits, device=cur_device) #first dimension for multi-GPU training
        accuracies = torch.zeros(1, self.opt.model_splits, device=cur_device) #first dimension for multi-GPU training

        if self.employ_autoregressive and self.calc_loss:
            c, loss[:, -1] = self.autoregressor(h)
        else:
            c = None

            if self.opt.model_splits == 1 and cur_loss is not None:
                loss[:, -1] = cur_loss
                accuracies[:, -1] = cur_accuracy

        return loss, c, h, accuracies


    def switch_calc_loss(self, calc_loss):
        ## by default models are set to not calculate the loss as it is costly
        ## this function can enable the calculation of the loss for training
        self.calc_loss = calc_loss
        if self.opt.model_splits == 1 and self.opt.loss == 0:
            self.autoregressor.calc_loss = calc_loss

        if self.opt.model_splits == 1 and self.opt.loss == 1:
            self.encoder[-1].calc_loss = calc_loss

        if self.opt.model_splits > 1:
            if self.opt.train_module == self.opt.model_splits:
                for i, layer in enumerate(self.encoder):
                    layer.calc_loss = calc_loss
            else:
                self.encoder[self.opt.train_module].calc_loss = calc_loss


def network(weight_file):
    opt = pickle.load(open('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/checkpoints/opt.pkl', 'rb'))
    print(opt)
    model = FullVisionModel(opt, True)
    #model = torch.nn.DataParallel(model)
    #data = torch.load('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/checkpoints/model_0_22.ckpt')
    data = torch.load(weight_file)
    #data = torch.load('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ExCRC/checkpoints/model_0_22.ckpt')
    new_state_dict = OrderedDict()
    #print('ffffffffffffff')
    for k, v in data.items():
        name = '0.' + k
        #name = k[:2] # remove `module.`
        #print(name, 11111111111111)
        #name = k.replace('model')
        new_state_dict[name] = v

    #print(data.keys())
    model.encoder.load_state_dict(new_state_dict)
    #print("'''''''''''''''")
    #for key in model.encoder.state_dict().keys():
        #print(key)
    model.model = None
    model.autoregressor = None
    #print(model)

    return model

#model = network()
#a = torch.Tensor(3, 1, 64, 64)

#print(model(a).shape)