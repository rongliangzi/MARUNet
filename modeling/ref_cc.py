import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils import *

class RefineModule(nn.Module):
    def __init__(self, in_ch, inc_ch):
        super(RefineModule, self).__init__()
        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)
        
        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        #self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        #self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        #self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        #self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        #self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        #self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        #self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        #self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        #self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1((self.conv1(hx)))
        hx = self.pool1(hx1)#1/2

        hx2 = self.relu2((self.conv2(hx)))#1/2
        hx = self.pool2(hx2)

        hx3 = self.relu3((self.conv3(hx)))#1/4
        hx = self.pool3(hx3)

        hx4 = self.relu4((self.conv4(hx)))#1/8
        hx = self.pool4(hx4)

        hx5 = self.relu5((self.conv5(hx)))#1/16

        hx = F.interpolate(hx5, size=hx4.shape[2:])#1/8

        d4 = self.relu_d4((self.conv_d4(torch.cat((hx,hx4),1))))#1/8
        hx = F.interpolate(d4, size=hx3.shape[2:])#1/4

        d3 = self.relu_d3((self.conv_d3(torch.cat((hx,hx3),1))))#1/4
        hx = F.interpolate(d3, size=hx2.shape[2:])#1/4

        d2 = self.relu_d2((self.conv_d2(torch.cat((hx,hx2),1))))#1/2
        hx = F.interpolate(d2, size=x.shape[2:])#1

        d1 = self.relu_d1((self.conv_d1(torch.cat((hx,hx1),1))))#1

        residual = self.conv_d0(d1)#1

        return x + residual
        
        
class RefCC(nn.Module):
    def __init__(self, load_model='', bn=False, NL='relu'):
        super(RefCC, self).__init__()
        self.bn = bn
        self.NL = NL
        self.front0 = make_layers([64, 64], in_channels=3, batch_norm=bn, NL=self.NL)
        self.front1 = make_layers(['M', 128, 128], in_channels=64, batch_norm=bn, NL=self.NL)
        self.front2 = make_layers(['M', 256, 256, 256], in_channels=128, batch_norm=bn, NL=self.NL)
        self.front3 = make_layers(['M', 512, 512, 512], in_channels=256, batch_norm=bn, NL=self.NL)
        self.front4 = make_layers(['M', 512, 512, 512], in_channels=512, batch_norm=bn, NL=self.NL)
        
        self.back4 = make_layers([512, 512], in_channels=512, dilation=True, batch_norm=bn, NL=self.NL)
        self.back3 = make_layers([512, 256], in_channels=512, dilation=True, batch_norm=bn, NL=self.NL)
        self.back2 = make_layers([256, 256], in_channels=256, dilation=True, batch_norm=bn, NL=self.NL)
        self.back1 = make_layers([256, 128], in_channels=256, dilation=True, batch_norm=bn, NL=self.NL)
        self.back0 = make_layers([64, 64], in_channels=128, dilation=True, batch_norm=bn, NL=self.NL)
        
        ## -------------Side Output--------------
        self.outconv4 = nn.Conv2d(512,1,3,padding=1)
        self.outconv3 = nn.Conv2d(256,1,3,padding=1)
        self.outconv2 = nn.Conv2d(256,1,3,padding=1)
        self.outconv1 = nn.Conv2d(128,1,3,padding=1)
        self.outconv0 = nn.Conv2d(64,1,3,padding=1)
        
        self.load_model = load_model
        self.refine = RefineModule(1, 64)
        self._init_weights()

    def forward(self, x_in):
        x0 = self.front0(x_in)#1 size
        x1 = self.front1(x0)#1/2 size
        x2 = self.front2(x1)#1/4 size
        x3 = self.front3(x2)#1/8 size
        x4 = self.front4(x3)#1/16 size
        
        xb4 = self.back4(x4)#1/16 size
        
        xb = F.interpolate(xb4, scale_factor=2)
        #
        xb3 = self.back3(xb)#1/8 size
        
        xb = F.interpolate(xb3, scale_factor=2)
        #
        xb2 = self.back2(xb3)#1/4 size
        
        xb = F.interpolate(xb2, scale_factor=2)
        
        xb1 = self.back1(xb)#1/2 size
        
        xb = F.interpolate(xb1, size=x_in.shape[2:])#1 size
        
        xb0 = self.back0(xb)#1 size
        
        d4 = self.outconv4(xb4)
        d4 = F.interpolate(d4, size=x_in.shape[2:])
        d3 = self.outconv3(xb3)
        d3 = F.interpolate(d3, size=x_in.shape[2:])
        d2 = self.outconv2(xb2)
        d2 = F.interpolate(d2, size=x_in.shape[2:])
        d1 = self.outconv1(xb1)
        d1 = F.interpolate(d1, size=x_in.shape[2:])
        d0 = self.outconv0(xb0)
        d0 = F.interpolate(d0, size=x_in.shape[2:])
        dout = self.refine(d0)
        return torch.abs(dout), torch.abs(d0), torch.abs(d1), torch.abs(d2), torch.abs(d3), torch.abs(d4)


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
            
            for i, (k, v) in enumerate(pretrained_model.items()):
                #print(i, k)
                
                if i < 4:
                    layer_id = 0
                    module_id = k.split('.')[-2]
                elif i < 8:
                    layer_id = 1
                    module_id = int(k.split('.')[-2]) - 4
                elif i < 14:
                    layer_id = 2
                    module_id = int(k.split('.')[-2]) - 9
                elif i < 20:
                    layer_id = 3
                    module_id = int(k.split('.')[-2]) - 16
                else:
                    break
                k = 'front' + str(layer_id) + '.' + str(module_id) + '.' + k.split('.')[-1]
                
                if k in model_dict and model_dict[k].size() == v.size():
                    print(k, ' parameters loaded!')
                    pretrained_dict[k] = v
            
            print(path, 'weights loaded!')
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            self.load_state_dict(torch.load(self.load_model))
            print(self.load_model,' loaded!')

