import torch.nn as nn
import torch
import torch.nn.functional as F
from .utils import conv_act


class MCNN(nn.Module):
    '''
    Multi-column CNN 
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''
    
    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        
        self.branch1 = nn.Sequential(conv_act( 3, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     conv_act(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     conv_act(32, 16, 7, same_padding=True, bn=bn),
                                     conv_act(16,  8, 7, same_padding=True, bn=bn))
        
        self.branch2 = nn.Sequential(conv_act( 3, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     conv_act(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     conv_act(40, 20, 5, same_padding=True, bn=bn),
                                     conv_act(20, 10, 5, same_padding=True, bn=bn))
        
        self.branch3 = nn.Sequential(conv_act( 3, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     conv_act(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     conv_act(48, 24, 3, same_padding=True, bn=bn),
                                     conv_act(24, 12, 3, same_padding=True, bn=bn))
        
        self.fuse = nn.Sequential(conv_act( 30, 1, 1, same_padding=True, bn=bn)) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1,x2,x3),1)
        x = self.fuse(x)
        #x = F.interpolate(x,size=im_data.shape[2:])
        return x
