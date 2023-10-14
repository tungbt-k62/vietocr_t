import torch
from torch import nn
import torch.nn.functional as F
import timm
import vietocr.model.backbone.vgg as vgg
from vietocr.model.backbone.resnet import Resnet50
import torchvision
from collections import OrderedDict

class CNN(nn.Module):
    def __init__(self, backbone, **kwargs):
        super(CNN, self).__init__()
        if backbone == 'vgg11_bn':
            self.model = vgg.vgg11_bn(**kwargs)
        elif backbone == 'vgg19_bn':
            self.model = vgg.vgg19_bn(**kwargs)
        elif backbone == 'resnet50':
            self.model = Resnet50(**kwargs)
        elif backbone == 'timm_backbone':
            # print('timm')
            self.model = timm.create_model(**kwargs)
            self.timm_chans = self.model.feature_info.channels()
            self.fpn = torchvision.ops.FeaturePyramidNetwork([self.timm_chans[2], self.timm_chans[3], self.timm_chans[4]], 256)
            # self.conv4 = nn.Conv2d(self.timm_chans[4], 256, 1)
            # self.conv3 = nn.Conv2d(self.timm_chans[3], 256, 1)
            # self.conv2 = nn.Conv2d(self.timm_chans[2], 256, 1)
        self.backbone = backbone

    def forward(self, x):
        if self.backbone == 'timm_backbone':
            xs =  self.model(x)
            # xs[4] = self.conv4(xs[4])
            # xs[4] = F.relu(xs[4])
            # xs[4] = F.interpolate(xs[4], size=xs[2].shape[2:], mode='bilinear', align_corners=False)
            
            # xs[3] = self.conv3(xs[3])
            # xs[3] = F.relu(xs[3])
            # xs[3] = F.interpolate(xs[3], size=xs[2].shape[2:], mode='bilinear', align_corners=False)

            # xs[2] = self.conv2(xs[2])
            # xs[2] = F.relu(xs[2])
            
            # x = xs[2]+xs[3]+xs[4]

            x = OrderedDict()
            x['feat2'] = xs[2]
            x['feat3'] = xs[3]
            x['feat4'] = xs[4]
            x = self.fpn(x)
            x = x['feat2']

            x = x.transpose(-1, -2)
            x = x.flatten(2)
            x = x.permute(-1, 0, 1)
            return x
        
        else:
            return self.model(x)

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != 'last_conv_1x1':
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
