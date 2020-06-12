import torch.nn as nn
import torch
from torchvision import models

from .utils import *

import torch.nn.functional as F


def initialize_weights(models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):    
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print( m )


class Res50(nn.Module):
    def __init__(self, pretrained=True, bn=True):
        super(Res50, self).__init__()
        self.bk1 = conv_act(1024, 128, 3, same_padding=True, NL='relu', bn=False)
        self.bk2 = conv_act(128, 128, 3, same_padding=True, NL='relu', bn=False)
        self.bk3 = conv_act(128, 64, 3, same_padding=True, NL='relu', bn=False)
        self.output_layer = conv_act(64, 1, 1, same_padding=True, NL='relu', bn=False)
        '''
        self.de_pred = nn.Sequential(conv_act(1024, 128, 1, same_padding=True, NL='relu'),
                                     conv_act(128, 1, 1, same_padding=True, NL='relu'))
        '''
        initialize_weights(self.modules())

        res = models.resnet50(pretrained=True)
        #res.load_state_dict(torch.load("/home/datamining/Models/resnet50-19c8e357.pth"))
        
        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 6, stride=1)        
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())

    def forward(self,x_in):
        
        x = self.frontend(x_in)

        x = self.own_reslayer_3(x) #1/8
        
        #x = self.de_pred(x)
        x = self.bk1(x)
        x = self.bk2(x)
        
        x = F.interpolate(x,size=x_in.shape[2:])
        x = self.bk3(x)
        x = self.output_layer(x)
        return x


def make_res_layer(block, planes, blocks, stride=1):

    downsample = None
    inplanes=512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)  


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out        