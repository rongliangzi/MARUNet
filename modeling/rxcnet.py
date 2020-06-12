'''ResNeXt as backbone to do crowd counting
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_layers
from torchvision import models
class Block(nn.Module):
    '''Grouped convolution block.'''
    expansion = 4

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1, bn=False):
        super(Block, self).__init__()
        self.bn=bn
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=not bn)
        
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=not bn)
        
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=not bn)
        if bn:
            self.bn1 = nn.BatchNorm2d(group_width)
            self.bn2 = nn.BatchNorm2d(group_width)
            self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            if bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*group_width)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
        else:
            out = F.relu(self.conv1(x))
            out = F.relu(self.conv2(out))
            out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
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

class ResXCountingNet(nn.Module):
    def __init__(self, num_blocks=[2,1,1], cardinality=16, bottleneck_width=4, bn=False):
        super(ResXCountingNet, self).__init__()
        self.bn = bn
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.de_pred = nn.Sequential(Conv2d(1024, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu'))
        self._init_weights()
        
        res = models.resnet50(pretrained=False)
        res.load_state_dict(torch.load("/home/datamining/Models/resnext50_32x4d-7cdf4587.pth"))
        
        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 6, stride=1)        
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())
        
        self.backend = make_layers([128], in_channels=1024, batch_norm=bn, dilation=True)
        self.output_layer = nn.Conv2d(128, 1, kernel_size=1)
        
    
    def _make_layer(self, num_blocks, stride):
        if not num_blocks:
            return nn.Sequential()
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.frontend(x)

        out = self.own_reslayer_3(out)
        out = self.backend(out)
        out = self.output_layer(out)
        out = F.interpolate(out, x.shape[2:])
        return out
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                #nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        pretrained_dict = dict()
        model_dict = self.state_dict()
        path = "/home/datamining/Models/resnext50_32x4d-7cdf4587.pth"
        pretrained_model = torch.load(path)
        # load the pretrained vgg16 parameters
        for k, v in pretrained_model.items():
            if k in model_dict and model_dict[k].size() == v.size():
                pretrained_dict[k] = v
                print(k)
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
