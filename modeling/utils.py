import torch.nn as nn
import torch
import torch.nn.functional as F


class conv_act(nn.Module):
    '''
    basic module for conv-(bn-)activation
    '''
    def __init__(self, in_channels, out_channels, kernel_size,  NL='relu', dilation=1, stride=1, same_padding=True, bn=False, se=False, groups=1):
        super(conv_act, self).__init__()
        padding = (kernel_size + (dilation - 1) * (kernel_size - 1) - 1) // 2 if same_padding else 0
        g = out_channels if groups == 0 else groups
        
        self.use_bn = bn
        if self.use_bn:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, groups=g, bias=False)
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, groups=g, bias=True)
        if NL == 'relu' :
            self.activation = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.activation = nn.PReLU() 
        elif NL == 'swish':
            self.activation = Swish()
        self.use_se = se
        if se:
            self.se = SEModule(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        if self.use_se:
            x = self.se(x)
        return x
        

class LSA(nn.Module):
    '''
    LocalSpatialAttention
    '''
    def __init__(self, kernel_size=7):
        super(LSA, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_size = x.shape[2:]
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        x = F.interpolate(x, size=x_size)
        return x
        
        
class ResBlock(nn.Module):
    '''
    residual block for conv-(bn-)activation
    '''
    def __init__(self, in_channels, kernel_size=3,  NL='relu', dilation=1, se=False, stride=1, same_padding=True, bn=False):
        super(ResBlock, self).__init__()
        padding = (kernel_size + (dilation - 1) * (kernel_size - 1) - 1) // 2 if same_padding else 0
        
        self.use_bn = bn
        if self.use_bn:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False)
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True)
        else:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=True)
        if NL == 'relu' :
            self.activation = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.activation = nn.PReLU() 
        elif NL == 'swish':
            self.activation = Swish()
        self.use_se = se
        if se:
            self.se = SEModule(out_channels)

    def forward(self, x_in):
        x = self.conv(x_in)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        if self.use_se:
            x = self.se(x)
        return x + x_in


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False, NL='relu', se=False, groups=1):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            g = v if groups==0 else groups
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate, groups=g)
            if NL=='prelu':
                nl_block = nn.PReLU()
            elif NL=='swish':
                nl_block = Swish()
            else:
                nl_block = nn.ReLU(inplace=True)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nl_block]
            else:
                layers += [conv2d, nl_block]
            if se:
                layers += [SEModule(v)]
            in_channels = v
    return nn.Sequential(*layers)


class DDCB(nn.Module):
    def __init__(self, in_channels):
        super(DDCB, self).__init__()
        self.conv1 = nn.Sequential(conv_act(in_channels, 256, 1), conv_act(256, 64, 3))
        self.conv2 = nn.Sequential(conv_act(in_channels+64, 256, 1), conv_act(256, 64, 3, dilation=2))
        self.conv3 = nn.Sequential(conv_act(in_channels+128, 256, 1), conv_act(256, 64, 3, dilation=3))
        self.conv4 = nn.Sequential(conv_act(in_channels+128, 512, 3))
    def forward(self, x):
        x1_raw = self.conv1(x)
        x1 = torch.cat([x, x1_raw], 1)
        x2_raw = self.conv2(x1)
        x2 = torch.cat([x, x1_raw, x2_raw], 1)
        x3_raw = self.conv3(x2)
        x3 = torch.cat([x, x2_raw, x3_raw], 1)
        output = self.conv4(x3)
        return output
        

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, NL='relu', se=False, dilation=1):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Sequential(conv_act(in_channels, 128, 1, NL=NL, se=se), conv_act(128, 128, 3, NL=NL, se=se, dilation=dilation))
        self.conv2 = nn.Sequential(conv_act(in_channels+128, 128, 1, NL=NL, se=se), conv_act(128, 128, 3, NL=NL, se=se, dilation=dilation))
        self.conv3 = nn.Sequential(conv_act(in_channels+256, 128, 1, NL=NL, se=se), conv_act(128, 128, 3, NL=NL, se=se, dilation=dilation))
        self.conv4 = nn.Sequential(conv_act(in_channels+384, out_channels, 1, NL=NL, se=se), conv_act(out_channels, out_channels, 3, NL=NL, se=se, dilation=dilation))
    def forward(self, x):
        x1_raw = self.conv1(x)
        x1 = torch.cat([x, x1_raw], 1)
        x2_raw = self.conv2(x1)
        x2 = torch.cat([x, x1_raw, x2_raw], 1)
        x3_raw = self.conv3(x2)
        x3 = torch.cat([x, x1_raw, x2_raw, x3_raw], 1)
        output = self.conv4(x3)
        return output
        
    
class DenseResBlock(nn.Module):
    def __init__(self, in_channels, NL='relu', se=False, dilation=1):
        super(DenseResBlock, self).__init__()
        self.conv1 = nn.Sequential(conv_act(in_channels, 128, 1, NL=NL, se=se), conv_act(128, 128, 3, NL=NL, se=se, dilation=dilation))
        self.conv2 = nn.Sequential(conv_act(in_channels+128, 128, 1, NL=NL, se=se), conv_act(128, 128, 3, NL=NL, se=se, dilation=dilation))
        self.conv3 = nn.Sequential(conv_act(in_channels+256, 128, 1, NL=NL, se=se), conv_act(128, 128, 3, NL=NL, se=se, dilation=dilation))
        self.conv4 = nn.Sequential(conv_act(in_channels+384, in_channels, 1, NL=NL, se=se))
    def forward(self, x):
        x1_raw = self.conv1(x)
        x1 = torch.cat([x, x1_raw], 1)
        x2_raw = self.conv2(x1)
        x2 = torch.cat([x, x1_raw, x2_raw], 1)
        x3_raw = self.conv3(x2)
        x3 = torch.cat([x, x1_raw, x2_raw, x3_raw], 1)
        output = self.conv4(x3) + x
        return output
        
        
class DilationPyramid(nn.Module):
    '''
    aggregate different dilations
    '''
    def __init__(self, in_channels, out_channels, dilations=[1,2,3,6], NL='relu', se=False):
        super(DilationPyramid, self).__init__()
        assert len(dilations)==4, 'length of dilations must be 4'
        self.conv1 = conv_act(in_channels, out_channels, 3, NL, dilation=dilations[0], se=se)
        self.conv2 = conv_act(in_channels, out_channels, 3, NL, dilation=dilations[1], se=se)
        self.conv3 = conv_act(in_channels, out_channels, 3, NL, dilation=dilations[2], se=se)
        self.conv4 = conv_act(in_channels, out_channels, 3, NL, dilation=dilations[3], se=se)
        #self.conv5 = conv_act(4*out_channels, 4*out_channels, 1, NL)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        output = torch.cat([x1, x2, x3, x4], 1)
        #output = self.conv5(output)
        return output


class SizePyramid(nn.Module):
    '''
    aggregate different filter sizes, [1,3,5,7] like ADCrowdNet's decoder
    '''
    def __init__(self, in_channels, out_channels, NL='relu', se=False):
        super(SizePyramid, self).__init__()
        self.conv1 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL, se=se), conv_act(out_channels, out_channels, 3, NL, se=se))
        self.conv2 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL, se=se), conv_act(out_channels, out_channels, 5, NL, se=se))
        self.conv3 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL, se=se), conv_act(out_channels, out_channels, 7, NL, se=se))
        self.conv4 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL, se=se))
        #self.conv5 = conv_act(4*out_channels, 4*out_channels, 1, NL)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        output = torch.cat([x1, x2, x3, x4], 1)
        #output = self.conv5(output)
        return output
        
        
class DepthPyramid(nn.Module):
    '''
    aggregate different depths, like TEDNet's decoder
    '''
    def __init__(self, in_channels, out_channels, NL='relu', se=False):
        super(DepthPyramid, self).__init__()
        self.conv1 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL, se=se), conv_act(out_channels, out_channels, 3, NL, se=se))
        self.conv2 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL, se=se), conv_act(out_channels, out_channels, 3, NL, se=se), conv_act(out_channels, out_channels, 3, NL, se=se))
        self.conv3 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL, se=se), conv_act(out_channels, out_channels, 3, NL, se=se), conv_act(out_channels, out_channels, 3, NL, se=se), conv_act(out_channels, out_channels, 3, NL, se=se))
        self.conv4 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL, se=se))
        self.conv5 = conv_act(4*out_channels, 4*out_channels, 1, NL)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        output = torch.cat([x1, x2, x3, x4], 1)
        output = self.conv5(output)
        return output
                
    
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SPModule(nn.Module):
    def __init__(self, in_channels, branch_out=None):
        super(SPModule, self).__init__()
        if not branch_out:
            # ensure the in and out have the same channels.
            branch_out = in_channels
        self.dilated1 = nn.Sequential(nn.Conv2d(in_channels, branch_out,3,padding=2, dilation=2),nn.ReLU(True))
        self.dilated2 = nn.Sequential(nn.Conv2d(in_channels, branch_out,3,padding=4, dilation=4),nn.ReLU(True))
        self.dilated3 = nn.Sequential(nn.Conv2d(in_channels, branch_out,3,padding=8, dilation=8),nn.ReLU(True))
        self.dilated4 = nn.Sequential(nn.Conv2d(in_channels, branch_out,3,padding=12, dilation=12),nn.ReLU(True))
        self.down_channels = nn.Sequential(nn.Conv2d(branch_out*4, in_channels,1),nn.ReLU(True))
    def forward(self,x):
        x1 = self.dilated1(x)
        x2 = self.dilated2(x)
        x3 = self.dilated3(x)
        x4 = self.dilated4(x)
        # concat
        x = torch.cat([x1,x2,x3,x4],1)
        x = self.down_channels(x)
        return x


class SEModule(nn.Module):
    def __init__(self, in_, reduction=16):
        super().__init__()
        squeeze_ch = in_//reduction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_, squeeze_ch, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(squeeze_ch, in_, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))