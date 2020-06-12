import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils import *


class M_VGG10(nn.Module):
    def __init__(self, load_model='', downsample=1, bn=False, NL='relu', objective='dmp', sp=False, se=False, block=''):
        super(M_VGG10, self).__init__()
        self.downsample = downsample
        self.bn = bn
        self.NL = NL
        self.objective = objective
        self.block = block
        self.features_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.features = make_layers(self.features_cfg, batch_norm=self.bn, NL=self.NL)
        self.sp = False
        if sp:
            print('use sp module')
            self.sp = True
            self.sp_module = SPModule(512)
        # basic cfg for backend is [512, 512, 512, 256, 128, 64]
        if not self.block:
            self.backconv1 = make_layers([512, 512, 512], in_channels=512, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.backconv1_ = make_layers([256], in_channels=512, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.backconv2 = make_layers([128], in_channels=256, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.backconv3 = make_layers([64], in_channels=128, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            
        elif self.block == 'dilation':
            print('use dilation pyramid in backend')
            self.backconv1 = nn.Sequential(DilationPyramid(512, 128), DilationPyramid(512, 128))
            self.backconv1_ = make_layers([256], in_channels=512, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.backconv2 = make_layers([128], in_channels=256, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.backconv3 = make_layers([64], in_channels=128, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            
        elif self.block == 'size':
            print('use size pyramid in backend')
            self.backconv1 = nn.Sequential(SizePyramid(512, 128), SizePyramid(512, 128))
            self.backconv1_ = make_layers([256], in_channels=512, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.backconv2 = make_layers([128], in_channels=256, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.backconv3 = make_layers([64], in_channels=128, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            
        elif self.block == 'depth':
            print('use depth pyramid in backend')
            self.backconv1 = nn.Sequential(DepthPyramid(512, 128), DepthPyramid(512, 128))
            self.backconv1_ = make_layers([256], in_channels=512, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.backconv2 = make_layers([128], in_channels=256, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.backconv3 = make_layers([64], in_channels=128, dilation=True, batch_norm=bn, NL=self.NL, se=se)
        
        elif self.block == 'Dense':
            self.backconv1 = DenseBlock(512, 512)
            self.backconv1_ = DenseBlock(512, 256)
            self.backconv2 = make_layers([128], in_channels=256, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.backconv3 =  make_layers([64], in_channels=128, dilation=True, batch_norm=bn, NL=self.NL, se=se)
        
        elif self.block == 'DenseRes':
            self.backconv1 = DenseResBlock(512)
            self.backconv1_ = DenseResBlock(512)
            self.backconv2 = make_layers([128], in_channels=512, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.backconv3 =  make_layers([64], in_channels=128, dilation=True, batch_norm=bn, NL=self.NL, se=se)
        
        elif self.block =='Res':
            self.backconv1 = nn.Sequential(ResBlock(512, dilation=2), ResBlock(512, dilation=2), ResBlock(512, dilation=2))
            self.backconv1_ = make_layers([256], in_channels=512, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.backconv2 = make_layers([128], in_channels=256, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.backconv3 =  make_layers([64], in_channels=128, dilation=True, batch_norm=bn, NL=self.NL, se=se)
        # objective is density map(dmp) and (binary) attention map(amp)
        if self.objective == 'dmp+amp':
            print('objective dmp+amp!')
            self.amp_process = make_layers([64, 64], in_channels=64, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.amp_layer = nn.Conv2d(64, 1, kernel_size=1)
            self.sgm = nn.Sigmoid()
        elif self.objective == 'dmp':
            print('objective dmp')
        else:
            raise Exception('objective must in [dmp, dmp+amp]')
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.load_model = load_model
        self._init_weights()

    def forward(self, x_in):
        x = self.features(x_in)
        
        if self.sp:
            x = self.sp_module(x)
        
        x = self.backconv1(x)
        if self.downsample == 4:
            x = F.interpolate(x, size=[s//4 for s in x_in.shape[2:]])
        elif self.downsample < 4:
            x = F.interpolate(x, scale_factor=2)
        #
        x = self.backconv1_(x)
        if self.downsample == 2:
            x = F.interpolate(x, size=[s//2 for s in x_in.shape[2:]])
        elif self.downsample < 2:
            x = F.interpolate(x, scale_factor=2)
        
        #
        x = self.backconv2(x)
        if self.downsample == 1:
            x = F.interpolate(x, size=x_in.shape[2:])
        
        x = self.backconv3(x)
        
        if self.objective == 'dmp+amp':
            dmp = self.output_layer(x)
            amp = self.amp_layer(self.amp_process(x))
            amp = self.sgm(amp)
            dmp = amp * dmp
            del x
            dmp = torch.abs(dmp)
            return dmp, amp
        else:
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
            
            self._random_init_weights()
            # load the pretrained vgg16 parameters
            for k, v in pretrained_model.items():
                if k in model_dict and model_dict[k].size() == v.size():
                    pretrained_dict[k] = v
                    print(k, ' parameters loaded!')
            
            print(path, 'weights loaded!')
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            self.load_state_dict(torch.load(self.load_model))
            print(self.load_model,' loaded!')

