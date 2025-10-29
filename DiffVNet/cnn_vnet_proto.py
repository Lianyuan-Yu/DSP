import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange 
from timm.layers import trunc_normal_  
import torch.distributed as dist  
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, td=True, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            #if not td:
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        #self.td = td
        #if self.td:
            #self.TD_LReLU = TD_LReLU(n_filters_in, n_filters_out)

    def forward(self, x):
        x = self.conv(x)
        #if self.td:
            #x = self.TD_LReLU(x)
        return x

class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x





class VNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters=16,
                 proj_dim=16, proto_mom=0.99, normalization='none', has_dropout=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout
        self.nclasses = n_classes

        
        self.prototypes = None
        self.proj_dim =proj_dim
        self.proto_mom = proto_mom  
        self.feat_norm = nn.LayerNorm(proj_dim)  

        self.block_one = ConvBlock(1, n_channels, n_filters, td=False, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, td=False, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, td=False, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, td=False, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, td=False, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, td=False, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, td=False, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, td=False, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, td=False, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        feature = x9
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out, feature

    def initialize_prototypes(self, feature_shape):

        _, C, D, H, W = feature_shape

        self.prototypes = nn.Parameter(
            torch.zeros(self.nclasses, C, D, H, W),
            requires_grad=False
        ).cuda()
        trunc_normal_(self.prototypes, std=0.02)
        print(f'Initialized prototypes with shape: {self.prototypes.shape}')

    def prototype_update(self, feature, label, proto_mom1=0.99):

        B, C, D, H, W = feature.shape

        if self.prototypes is None:
            self.initialize_prototypes(feature.shape)
            #print(self.prototypes)

        new_protos = torch.zeros_like(self.prototypes)
        counts = torch.zeros(self.nclasses, 1, D, H, W, device=feature.device)


        for k in range(self.nclasses):
            mask = (label == k).float()
            count = mask.sum(dim=0, keepdim=True)
            counts[k] = count
            weighted_sum = (feature * mask).sum(dim=0)
            with torch.no_grad():
                if count.sum() > 0:
                    current_proto = weighted_sum / (count + 1e-8)
                    new_protos[k] = proto_mom1 * self.prototypes[k] + (1 - proto_mom1) * current_proto
                else:
                    new_protos[k] = self.prototypes[k]
        self.prototypes.data = F.normalize(new_protos, p=2, dim=1)


        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes.data = protos

    def forward(self, input, label=None, use_prototype=True):
        features = self.encoder(input)
        classifier, feature = self.decoder(features) # feature形状 [B, C, D, H, W]

        B, C, D, H, W = feature.shape
        feat_flat = rearrange(feature, "b c d h w -> (b d h w) c")
        feat_flat = self.feat_norm(feat_flat)
        feat_flat = F.normalize(feat_flat, p=2, dim=-1)
        feature = rearrange(feat_flat, "(b d h w) c -> b c d h w", b=B, d=D, h=H, w=W)

        if use_prototype:
            if label is not None:  
                self.prototype_update(feature, label)  
            else:  
                pseudo_label = torch.argmax(classifier, dim=1)  # [B, D, H, W]
                self.prototype_update(feature, pseudo_label.unsqueeze(dim=1))

        proto_ck = self.prototypes

        return feature, classifier, proto_ck
