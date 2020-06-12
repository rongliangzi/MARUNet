'''EfficientNet Architecture
https://github.com/zsef123/EfficientNets-PyTorch
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_act(in_, out_, kernel_size, stride=1, groups=1, bias=True):
    conv2d = nn.Conv2d(in_, out_, kernel_size, stride, padding=(kernel_size-1)//2, groups=groups, bias=bias)
    '''
    nn.init.normal_(conv2d.weight, std=0.1)
    if conv2d.bias is not None:
        nn.init.constant_(conv2d.bias, 0)
       ''' 
    return nn.Sequential(conv2d,
        #nn.ReLU(True),
        Swish()
    )

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEModule(nn.Module):
    def __init__(self, in_, squeeze_ch):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_, squeeze_ch, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(squeeze_ch, in_, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))

class MBConv(nn.Module):
    def __init__(self, in_, out_, expand, kernel_size, stride, skip, se_ratio):
        super().__init__()
        mid_ = in_ * expand
        self.expand_conv = conv_act(in_, mid_, kernel_size=1) if expand != 1 else nn.Identity()

        self.depth_wise_conv = conv_act(mid_, mid_, kernel_size=kernel_size, stride=stride, groups=mid_)

        self.se = SEModule(mid_, int(in_ * se_ratio)) if se_ratio > 0 else nn.Identity()

        self.project_conv = nn.Sequential(nn.Conv2d(mid_, out_, kernel_size=1, stride=1))
        
        self.skip = skip and (stride == 1) and (in_ == out_)
        
        self.dropconnect = nn.Identity()
        
    def forward(self, inputs):
        #print('inputs:',inputs.sum())
        expand = self.expand_conv(inputs)
        #print('expand:',expand.sum())
        x = self.depth_wise_conv(expand)
        #print('depth wise:',x.sum())
        x = self.se(x)
        x = self.project_conv(x)
        if self.skip:
            x = self.dropconnect(x)
            x = x + inputs
        return x

class MBBlock(nn.Module):
    def __init__(self, in_, out_, expand, kernel, stride, num_repeat, skip, se_ratio):
        super().__init__()
        layers = [MBConv(in_, out_, expand, kernel, stride, skip, se_ratio,)]
        for i in range(1, num_repeat):
            layers.append(MBConv(out_, out_, expand, kernel, 1, skip, se_ratio))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Efficient Counting Net
class ECNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = conv_act(3, 32, kernel_size=3, stride=2)
        
        def renew_ch(x):
            return x
        def renew_repeat(x):
            return x
        expand = 4
        self.blocks = nn.Sequential(
            #       input channel  output    expand  k  s                   skip  se
            MBBlock(renew_ch(32), renew_ch(16), 1, 3, 1, renew_repeat(1), True, 0.25, ),
            MBBlock(renew_ch(16), renew_ch(24), expand, 3, 2, renew_repeat(2), True, 0.25, ),
            MBBlock(renew_ch(24), renew_ch(40), expand, 5, 1, renew_repeat(2), True, 0.25, ),
            MBBlock(renew_ch(40), renew_ch(80), expand, 3, 2, renew_repeat(3), True, 0.25, ),#2->1
            MBBlock(renew_ch(80), renew_ch(112), expand, 5, 1, renew_repeat(3), True, 0.25, ),
            MBBlock(renew_ch(112), renew_ch(192), expand, 5, 1, renew_repeat(4), True, 0.25, ),#2->1
            MBBlock(renew_ch(192), renew_ch(320), expand, 3, 1, renew_repeat(1), True, 0.25, )
        )
        self.back1 = conv_act(320, 256, 1)
        self.back2 = conv_act(256, 64, 1)
        self.back3 = conv_act(64, 1, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            self.real_init_weights(m)
    def real_init_weights(self,m):
        if isinstance(m, list):
            for mini_m in m:
                self.real_init_weights(mini_m)
        else:
            if isinstance(m, nn.Conv2d):    
                #nn.init.kaiming_normal_(m.weight)
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Module):
                for mini_m in m.children():
                    self.real_init_weights(mini_m)
            else:
                print( m )

    def forward(self, inputs):
        x = self.stem(inputs)
        #print(x.sum())
        x = self.blocks(x)
        #print(x.sum(), x[0].sum())
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)
        x = self.back1(x)
        #print(x.sum())
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)
        x = self.back2(x)
        #print(x.sum())
        x = F.interpolate(x,size=inputs.shape[-2:],mode='bilinear',align_corners=True)
        x = self.back3(x)
        #print(x.sum())
        return x

if __name__ == "__main__":
    print("Efficient B0 Summary")
    net = ECNet()
    y = net(torch.randn(1,3,256,256))
    print(y.sum())
    from torchsummary import summary
    summary(net.cuda(), (3, 224, 224))