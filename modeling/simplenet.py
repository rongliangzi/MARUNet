import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils import make_layers


class SimNet(nn.Module):
    def __init__(self):
        super(SimNet, self).__init__()
        
        self.features_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 256, 256]
        self.features = make_layers(self.features_cfg, batch_norm=bn)
        
        self.backconv1_cfg = [128, 128]
        self.backconv1 = make_layers(self.backconv1_cfg, in_channels=256, dilation=True, batch_norm=bn)
        self.backconv2 = nn.Conv2d(128, 32, 3, dilation=2, padding=2)
        
        self.backconv3 = nn.Conv2d(32, 32, 3, dilation=2, padding=2)
        
        self.output_layer = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x_in):
        x = self.features(x_in)
        x = self.backconv1(x)
        
        x = F.relu(self.backconv2(x)) 
        
        x = F.relu(self.backconv3(x))
        
        x = self.output_layer(x)
        
        return x

