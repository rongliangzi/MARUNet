import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils import *

class PlusNet(nn.Module):
    def __init__(self, load_model='', downsample=1, bn=False, NL='relu', objective='dmp', sp=False, se=False, block=''):
        super(PlusNet, self).__init__()
        self.downsample = downsample
        self.bn = bn
        self.NL = NL
        self.objective = objective
        self.front0 = make_layers([64, 64], in_channels=3, batch_norm=bn, NL=self.NL)
        self.front1 = make_layers(['M', 128, 128], in_channels=64, batch_norm=bn, NL=self.NL)
        self.front2 = make_layers(['M', 256, 256, 256], in_channels=128, batch_norm=bn, NL=self.NL)
        self.front3 = make_layers(['M', 512, 512, 512], in_channels=256, batch_norm=bn, NL=self.NL)
        self.front4 = make_layers(['M', 512, 512, 512], in_channels=512, batch_norm=bn, NL=self.NL)
        
        self.brg = make_layers([512], in_channels=512, dilation=True, batch_norm=bn, NL=self.NL, se=se)
        self.back4 = make_layers([512], in_channels=1536, dilation=True, batch_norm=bn, NL=self.NL, se=se)
        self.back3 = make_layers([256,], in_channels=1280, dilation=True, batch_norm=bn, NL=self.NL, se=se)
        self.back2 = make_layers([128], in_channels=640, dilation=True, batch_norm=bn, NL=self.NL, se=se)
        self.back1 = make_layers([64], in_channels=320, dilation=True, batch_norm=bn, NL=self.NL, se=se)
        self.back0 = make_layers([64], in_channels=128, dilation=True, batch_norm=bn, NL=self.NL, se=se)
        
        
        # objective is density map(dmp) and (binary) attention map(amp)
        if self.objective == 'dmp+amp':
            print('objective dmp+amp!')
            self.amp_process = make_layers([64,64], in_channels=64, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.amp_layer = nn.Conv2d(64, 1, kernel_size=1)
            self.sgm = nn.Sigmoid()
        elif self.objective == 'dmp':
            print('objective dmp')
        else:
            raise Exception('objective must in [dmp, dmp+amp]')
        
        self.outconvb = nn.Conv2d(512,1,3,padding=1)
        self.outconv4 = nn.Conv2d(512,1,3,padding=1)
        self.outconv3 = nn.Conv2d(256,1,3,padding=1)
        self.outconv2 = nn.Conv2d(128,1,3,padding=1)
        self.outconv1 = nn.Conv2d(64,1,3,padding=1)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.load_model = load_model
        self._init_weights()

    def forward(self, x_in):
        x0 = self.front0(x_in)#1 size, 64
        x1 = self.front1(x0)#1/2 size, 128
        x2 = self.front2(x1)#1/4 size, 256
        x3 = self.front3(x2)#1/8 size, 512
        x4 = self.front4(x3)#1/16 size, 512
        x_brg = self.brg(x4)#1/16 size, 512
        
        x3d = F.interpolate(x3, size=x4.shape[2:])
        xb4 = torch.cat([x_brg, x4, x3d], 1)#1/16, 1536
        xb4 = self.back4(xb4) #1/16 size, 512
        
        xb3 = F.interpolate(xb4, size=[x_in.shape[2]//8, x_in.shape[3]//8]) #1/8 size, 512
        x2d = F.interpolate(x2, size=x3.shape[2:])
        xb3 = torch.cat([x3, xb3, x2d], dim=1) #1/8 size, 1280
        xb3 = self.back3(xb3) #1/8 size, 256
        
        xb2 = F.interpolate(xb3, size=[x_in.shape[2]//4, x_in.shape[3]//4]) #1/4 size, 256
        x1d = F.interpolate(x1, x2.shape[2:])
        xb2 = torch.cat([x2, xb2, x1d], dim=1) #1/4 size, 640
        xb2 = self.back2(xb2) #1/4 size, 128
        
        xb1 = F.interpolate(xb2, size=[x_in.shape[2]//2, x_in.shape[3]//2]) #1/2 size, 128
        x0d = F.interpolate(x0, x1.shape[2:])
        xb1 = torch.cat([x1, xb1, x0d], dim=1)#1/2, 320
        xb1 = self.back1(xb1)#1/2, 64
        
        xb0 = F.interpolate(xb1, size=x_in.shape[2:]) #1 size, 64
        xb0 = torch.cat([x0, xb0], 1)#1,128
        xb0 = self.back0(xb0)#1, 64
        
        x_brg = F.interpolate(x_brg, size=x_in.shape[2:])
        db = self.outconvb(x_brg)
        xb4 = F.interpolate(xb4, size=x_in.shape[2:])
        d4 = self.outconv4(xb4)
        xb3 = F.interpolate(xb3, size=x_in.shape[2:])
        d3 = self.outconv3(xb3)
        xb2 = F.interpolate(xb2, size=x_in.shape[2:])
        d2 = self.outconv2(xb2)
        xb1 = F.interpolate(xb1, size=x_in.shape[2:])
        d1 = self.outconv1(xb1)
        
        
        if self.objective == 'dmp+amp':
            dmp = self.output_layer(xb0)
            amp = self.amp_layer(self.amp_process(xb0))
            amp = self.sgm(amp)
            dmp = amp * dmp
            d1 = amp * d1
            d2 = amp * d2
            d3 = amp * d3
            d4 = amp * d4
            db = amp * db
            return torch.abs(dmp), torch.abs(d1), torch.abs(d2), torch.abs(d3), torch.abs(d4), torch.abs(db), amp
        else:
            x = self.output_layer(xb0)
            #x_ref = self.refine(x)
            return torch.abs(x),torch.abs(d1), torch.abs(d2), torch.abs(d3), torch.abs(d4), torch.abs(db)

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
            path = '/home/datamining/Models/vgg16-397923af.pth'
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
                elif i < 26:
                    layer_id = 4
                    module_id = int(k.split('.')[-2]) - 23
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

