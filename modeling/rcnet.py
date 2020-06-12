'''ResNet as backbone to do crowd counting
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
from torchvision import models
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation=dilation, groups=1)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


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


class ResCountingNet(nn.Module):
    def __init__(self):
        super(ResCountingNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        
        self.back1 = conv_act(1024, 256, 1, same_padding=True, NL='relu', bn=False)
        self.back2 = conv_act(256, 256, 1, same_padding=True, NL='relu', bn=False)
        self.back3 = conv_act(256, 1, 1, same_padding=True, NL='relu', bn=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        res = models.resnet50(pretrained=False)
        '''
        for name, module in res.named_children():
            print('resnet50 children module:', name)
        for name,_ in self.named_modules():
            print('self:', name)
        '''
        res.load_state_dict(torch.load("/home/datamining/Models/resnet50-19c8e357.pth"))
        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.layer1, res.layer2, res.layer3
        )
        
    def forward(self, x):
        x = self.frontend(x)
        #print(x.size())
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)
        x = self.back1(x)
        
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)
        x = self.back2(x)
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)
        x = self.back3(x)
        
        return x

if __name__=='__main__':
    net=ResCountingNet()
    y = net(torch.randn(1,3,224,224))
    print(y.size())
    from torchsummary import summary
    summary(net.cuda(), (3, 224, 224))