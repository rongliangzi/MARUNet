import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils import make_layers


class PIPModule4(nn.Module):
    """
    Pyramid in Pyramid Module, 2*2=4 branches
    """
    def __init__(self, in_channels=512):
        super(PIPModule4, self).__init__()
        self.size1_dilation1 = nn.Conv2d(in_channels, in_channels//4,3,padding=2, dilation=2)
        self.size1_dilation2 = nn.Conv2d(in_channels, in_channels//4,3,padding=4, dilation=4)
        self.size2_dilation1 = nn.Conv2d(in_channels, in_channels//4,3,padding=2, dilation=2)
        self.size2_dilation2 = nn.Conv2d(in_channels, in_channels//4,3,padding=4, dilation=4)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        #self.conv = nn.Conv2d(in_channels, in_channels,kernel_size=1)
    def forward(self,x_in):
        x1 = self.relu(self.size1_dilation1(x_in))
        x2 = self.relu(self.size1_dilation2(x_in))
        x_down = self.pool(x_in)
        x3 = F.interpolate(self.relu(self.size2_dilation1(x_down)), size=x1.shape[2:])
        x4 = F.interpolate(self.relu(self.size2_dilation2(x_down)), size=x1.shape[2:])
        x = torch.cat([x1,x2,x3,x4], 1)
        #x = self.conv(x)
        return x
        


class PIPNet(nn.Module):
    """
    Pyramid in Pyramid Network
    """
    def __init__(self):
        super(PIPNet, self).__init__()
        self.features_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.features = make_layers(self.features_cfg)
        self.pip = PIPModule4(512)
        self.backconv_cfg = [512, 512, 256, 128, 64]
        self.backconv = make_layers(self.backconv_cfg, in_channels=512, dilation=True)
        
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self._init_weights()
    def forward(self,x):
        x = self.features(x)
        x = self.pip(x)
        x = self.backconv(x)
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