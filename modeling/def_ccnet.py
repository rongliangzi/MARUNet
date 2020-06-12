import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils import *
from .deform_conv_v2 import *


# deformable crowd counting net. The front-end is vgg16, and the back-end uses deformable conv v2.
class DefCcNet(nn.Module):
    def __init__(self, load_model='', downsample=1, bn=False, NL='relu', objective='dmp', sp=False):
        super(DefCcNet, self).__init__()
        self.downsample = downsample
        self.bn = bn
        self.NL = NL
        self.objective = objective
        self.features_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.features = make_layers(self.features_cfg, batch_norm=bn, NL=self.NL)
        self.sp = False
        if sp:
            self.sp = True
            self.sp_module = SPModule(512)
        '''
        in_ = 512
        back_ = []
        for out_ in [512, 512, 512]:
            back_.append(DeformConv2d(in_, out_, kernel_size=3, padding=1, bias=True, modulation=True))
            back_.append(nn.ReLU(inplace=True))
            in_ = out_
        self.backconv1 = nn.Sequential(*back_)
        '''
        self.backconv1_cfg = [512, 512, 512]
        self.backconv1 = make_layers(self.backconv1_cfg, in_channels=512, dilation=True, batch_norm=bn, NL=self.NL)
        
        self.backconv1_ = nn.Sequential(DeformConv2d(inc=512, outc=256, kernel_size=3, padding=1, bias=True, modulation=True), 
                                        nn.ReLU(inplace=True))
        self.backconv2 = nn.Sequential(DeformConv2d(inc=256, outc=128, kernel_size=3, padding=1, bias=True, modulation=True), 
                                        nn.ReLU(inplace=True))
        
        self.backconv3 = nn.Sequential(DeformConv2d(inc=128, outc=64, kernel_size=3, padding=1, stride=1, bias=True, modulation=True), 
                                        nn.ReLU(inplace=True))
        # objective is density map(dmp) and (binary) attention map(amp)
        if self.objective == 'dmp+amp':
            print('objective dmp+amp!')
            self.amp_layer = nn.Conv2d(64, 1, kernel_size=1)
            self.sgm = nn.Sigmoid()
        elif self.objective == 'dmp':
            print('objective dmp')
        else:
            raise Exception('objective must in [dmp, dmp+amp]')
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.load_model = load_model
        self._init_weights()
        #self._random_init_weights()

    def forward(self, x_in):
        x = self.features(x_in)
        
        x = self.backconv1(x)
        
        if self.downsample==8:
            x = F.interpolate(x, size=[s//8 for s in x_in.shape[2:]])
        elif self.downsample<8:
            x = F.interpolate(x, scale_factor=2)
        #
        x = self.backconv1_(x)
        
        if self.downsample==4:
            x = F.interpolate(x, size=[s//4 for s in x_in.shape[2:]])
        elif self.downsample<4:
            x = F.interpolate(x, scale_factor=2)
        #
        x = self.backconv2(x)
        
        if self.downsample==2:
            x = F.interpolate(x, size=[s//2 for s in x_in.shape[2:]])
        elif self.downsample<2:
            x = F.interpolate(x, scale_factor=2)
        
        x = self.backconv3(x)
        
        if self.downsample==1:
            x = F.interpolate(x, size=x_in.shape[2:])
        
        
        if self.objective == 'dmp+amp':
            dmp = self.output_layer(x)
            amp = self.amp_layer(x)
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
            self.load_state_dict(torch.load(self.load_model))

