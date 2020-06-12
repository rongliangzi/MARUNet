import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils import make_layers

class SPModule(nn.Module):
    def __init__(self, in_channels, branch_out=128):
        super(SPModule, self).__init__()
        self.dilated1 = nn.Sequential(nn.Conv2d(in_channels, branch_out,3,padding=2, dilation=2),nn.ReLU(True))
        self.dilated2 = nn.Sequential(nn.Conv2d(in_channels, branch_out,3,padding=4, dilation=4),nn.ReLU(True))
        self.dilated3 = nn.Sequential(nn.Conv2d(in_channels, branch_out,3,padding=8, dilation=8),nn.ReLU(True))
        self.dilated4 = nn.Sequential(nn.Conv2d(in_channels, branch_out,3,padding=12, dilation=12),nn.ReLU(True))
    def forward(self,x):
        x1 = self.dilated1(x)
        x2 = self.dilated2(x)
        x3 = self.dilated3(x)
        x4 = self.dilated4(x)
        # concat
        x = torch.cat([x1,x2,x3,x4],1)
        return x
class SPNet(nn.Module):
    """
    Scale Pyramid Network for Crowd Counting
    """
    def __init__(self):
        super(SPNet, self).__init__()
        self.features_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.features = make_layers(self.features_cfg)
        self.SPM = SPModule(512, branch_out=128)
        self.backend_cfg = [512]*3+[256]
        self.backend = make_layers(self.backend_cfg, in_channels=512)
        self.output_layer = nn.Conv2d(256,1,1)
        #self.output_layer = nn.Sequential(nn.Conv2d(256,1,1),nn.ReLU(True))
        self._init_weights()
    def forward(self,x):
        x = self.features(x)
        x = self.SPM(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    def _random_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def _init_weights(self):
        pretrained_dict = dict()
        model_dict = self.state_dict()
        path = '/home/datamining/Models/vgg16-397923af.pth'
        pretrained_model = torch.load(path)
        self._random_init_weights()
        # load the pretrained vgg16 parameters
        for k, v in pretrained_model.items():
            if k in model_dict and model_dict[k].size() == v.size():
                pretrained_dict[k] = v
                print(k)
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)