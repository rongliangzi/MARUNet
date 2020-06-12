import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils import make_layers


class VGG_UNet(nn.Module):
    def __init__(self, upsample=True, bn=False, NL='relu'):
        super(VGG_UNet, self).__init__()
        self.upsample = upsample
        self.bn = bn
        self.NL = NL
        # config
        self.front_conv1_cfg = [64, 64]
        self.front_conv2_cfg = ['M', 128, 128]
        self.front_conv3_cfg = ['M', 256, 256, 256]
        self.front_conv4_cfg = ['M', 512, 512, 512]
        self.front_conv5_cfg = ['M', 512, 512, 512]

        self.dmp_conv1_cfg = [(256, 1), (256, 3)]
        self.dmp_conv2_cfg = [(128, 1), (128, 3)]
        self.dmp_conv3_cfg = [(64, 1), (64, 3), (32, 3)]
        
        # layers
        self.front_conv1 = make_layers(self.front_conv1_cfg, batch_norm=True)
        self.front_conv2 = make_layers(self.front_conv2_cfg, in_channels=64, batch_norm=True)
        self.front_conv3 = make_layers(self.front_conv3_cfg, in_channels=128, batch_norm=True)
        self.front_conv4 = make_layers(self.front_conv4_cfg, in_channels=256, batch_norm=True)
        self.front_conv5 = make_layers(self.front_conv5_cfg, in_channels=512, batch_norm=True)

        self.dmp_conv1 = make_back_layers(self.dmp_conv1_cfg, in_channels=1024, batch_norm=True)
        self.dmp_conv2 = make_back_layers(self.dmp_conv2_cfg, in_channels=512, batch_norm=True)
        self.dmp_conv3 = make_back_layers(self.dmp_conv3_cfg, in_channels=256, batch_norm=True)

        self.amp_last = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())
        self.out_layer = nn.Conv2d(32, 1, kernel_size=1)
        
        
        self.load_model = load_model
        self._init_weights()
        #self._random_init_weights()

    def forward(self, x_in):
        x1 = self.front_conv1(x_in)
        x2 = self.front_conv2(x_1)
        x3 = self.front_conv3(x_2)
        x4 = self.front_conv4(x_3)
        x5 = self.front_conv5(x_4)
        
        x = self.backconv1(x)
        
        if self.upsample:
          x = F.interpolate(x, scale_factor=2)
        #
        x = self.backconv1_(x)
        
        if self.upsample:
          x = F.interpolate(x, scale_factor=2)
        #
        x = self.backconv2(x)
        
        if self.upsample:
          x = F.interpolate(x, scale_factor=2)
        
        x = self.backconv3(x)
        
        if self.upsample:
          x = F.interpolate(x, size=x_in.shape[2:])
        
        x = self.output_layer(x)
        
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
        self_dict = self.state_dict()
        pretrained_dict = dict()
        # load the weights of vgg16_bn
        model_vgg16bn = torch.load('/unsullied/sharefs/rongliangzi/isilon-home/models/vgg16_bn-6c64b313.pth')
        module_dict = [0]*2+[1]*4+[3]*2+[4]*4+[6]*2+[7]*4
        for i, (k, v) in enumerate(model_vgg16bn.items()):
            if i < 12:
                layer_id = 1
                module_id = module_dict[i]
            elif i < 24:
                layer_id = 2
                module_id = module_dict[i-12]
            elif i < 42:
                layer_id = 3
                module_id = module_dict[i-24]
            elif i < 60:
                layer_id = 4
                module_id = module_dict[i-42]
            elif i < 78:
                layer_id = 5
                module_id = module_dict[i-60]
            else:
                    break
            k = 'conv'+str(layer_id)+'.'+str(module_id)+'.'+k.split('.')[-1]
            if k in self_dict and self_dict[k].size == v.size:
                print(k)
                pretrained_dict[k] = v
            print('load weights of vgg16bn')
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

