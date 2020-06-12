import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils import *


class CSRNet(nn.Module):
    def __init__(self, load_model='', downsample=1, bn=False):
        super(CSRNet, self).__init__()
        self.downsample = downsample
        self.bn = bn
        self.features_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.features_cfg)
        self.backend_cfg = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_cfg, in_channels=512, dilation=True)
       
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.load_model = load_model
        self._init_weights()

    def forward(self, x_in):
        x = self.frontend(x_in)
        
        x = self.backend(x)
        
        x = self.output_layer(x)
        x = torch.abs(x)
        return x

    def _random_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _init_weights(self):
        if not self.load_model:
            pretrained_dict = dict()
            model_dict = self.state_dict()
            path = "/home/datamining/Models/vgg16_bn-6c64b313.pth" if self.bn else '/home/datamining/Models/vgg16-397923af.pth'
            pretrained_model = torch.load(path)
            print(path,' loaded!')
            self._random_init_weights()
            # load the pretrained vgg16 parameters
            for k, v in pretrained_model.items():
                if k in model_dict and model_dict[k].size() == v.size():
                    pretrained_dict[k] = v
                    print(k, ' parameters loaded!')
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            ckpt = torch.load(self.load_model)
            if self.load_model[-4:]=='.tar':
                self.load_state_dict(ckpt['state_dict'])
            else:
                self.load_state_dict(ckpt)
            print(self.load_model,' loaded!')

