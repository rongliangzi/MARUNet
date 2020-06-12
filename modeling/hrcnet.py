import torch
import torch.nn as nn
from .utils import *
import torch.nn.functional as F


class DualUp(nn.Module):
    def __init__(self, in_, out_, size=None):
        super(DualUp, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_, out_//2, 3, stride=1, padding=1, dilation=1), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_, out_//2, 3, stride=1, padding=1, dilation=1), nn.ReLU(True))
    def forward(self, x):
        x1 = F.interpolate(x, scale_factor=2)
        x1 = self.conv1(x1)
        x2 = self.conv2(x)
        x2 = F.interpolate(x2, scale_factor=2)
        output = torch.cat([x1,x2],1)
        return output

class HRCNetV1(nn.Module):
    def __init__(self):
        super(HRCNetV1, self).__init__()
        self.c1_1 = nn.Sequential(nn.Conv2d(3, 32, 3, stride=1, padding=1, dilation=1), nn.ReLU(True),)
        self.c1_2 = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1, dilation=1), nn.ReLU(True))
        self.c2_1 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=1, padding=1, dilation=1), nn.ReLU(True))
        self.c2_2 = nn.Sequential(nn.Conv2d(32, 32, 3, stride=2, padding=1, dilation=1), nn.ReLU(True))
        self.c2_3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=4, padding=1, dilation=1), nn.ReLU(True))
        self.c2_4 = nn.Sequential(nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=1), nn.ReLU(True))
        self.c2_5 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1, dilation=1), nn.ReLU(True))
        
        self.c3_1 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=4, padding=1, dilation=1), nn.ReLU(True))
        self.c3_2 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1, dilation=1), nn.ReLU(True))
        self.c3_3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.ReLU(True))
        
        self.mid = make_layers([128]*4+[256]*2, in_channels=256, dilation=True, batch_norm=False, NL='relu')
        self.back1 = DualUp(256,256)
        self.back2 = DualUp(256,128)
        self.back3_1 = nn.Sequential(nn.Conv2d(128, 32, 3, stride=1, padding=1, dilation=1), nn.ReLU(True))
        self.back3_2 = nn.Sequential(nn.Conv2d(128, 32, 3, stride=1, padding=1, dilation=1), nn.ReLU(True))
        self.output_layer = nn.Sequential(nn.Conv2d(128,1,1), nn.ReLU(True))
        self.init_weights()
    def forward(self,x_in):
        x1_1 = self.c1_1(x_in)#1
        x1_2 = self.c1_2(x_in)#//2
        x2_1 = self.c2_1(x1_1)
        x2_2 = torch.cat([self.c2_2(x1_1), self.c2_4(x1_2)],1)#//2
        x2_3 = torch.cat([self.c2_3(x1_1), self.c2_5(x1_2)],1)#//4
        
        x3 = torch.cat([self.c3_1(x2_1), self.c3_2(x2_2), self.c3_3(x2_3)],1)#//4
        x_mid = self.mid(x3)
        x_up_1 = self.back1(x_mid)#//4*2
        x_up_2 = self.back2(x_up_1)#//4*4
        x_up_3_1 = F.interpolate(x_up_2, size=x_in.shape[2:])
        x_up_3_1 = self.back3_1(x_up_3_1)
        x_up_3_2 = self.back3_2(x_up_2)
        x_up_3_2 = F.interpolate(x_up_3_2, size=x_in.shape[2:])
        x_up_3 = torch.cat([x_up_3_1, x_up_3_2, x2_1],1)#original size
        output = self.output_layer(x_up_3) 
        return output
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):    
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #print('nn.Conv2d')
                print(m)
                #nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Module):
                pass
                #print('nn.Module')
                #print(m)
            else:
                print( 'else' )


class HRCNetV3(nn.Module):
    def __init__(self):
        super(HRCNetV3, self).__init__()
        self.features_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,'M', 512, 512, 512]
        self.features = make_layers(self.features_cfg)
        
        self.back1 = DualUp(512,256)
        self.back2 = DualUp(256,128)
        self.back3 = DualUp(128,64)
        self.back3_1 = nn.Sequential(nn.Conv2d(64, 32, 3, stride=1, padding=1, dilation=1), nn.ReLU(True))
        self.back3_2 = nn.Sequential(nn.Conv2d(64, 32, 3, stride=1, padding=1, dilation=1), nn.ReLU(True))
        self.output_layer = nn.Sequential(nn.Conv2d(64,1,1), nn.ReLU(True))
        self.init_weights()
    def forward(self,x_in):
        x = self.features(x_in)
        
        x = self.back1(x)#//4*2
        x = self.back2(x)#//4*4
        x = self.back3(x)
        x_up_1 = F.interpolate(x, size=x_in.shape[2:])
        x_up_1 = self.back3_1(x_up_1)
        x_up_2 = self.back3_2(x)
        x_up_2 = F.interpolate(x_up_2, size=x_in.shape[2:])
        x_up_3 = torch.cat([x_up_1, x_up_2],1)#original size
        output = self.output_layer(x_up_3) 
        return output
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):    
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #print('nn.Conv2d')
                print(m)
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Module):
                pass
                #print('nn.Module')
                #print(m)
            else:
                print( 'else' )
        pretrained_dict = dict()
        model_dict = self.state_dict()
        path = '/home/datamining/Models/vgg16-397923af.pth'
        pretrained_model = torch.load(path)
        # load the pretrained vgg16 parameters
        for k, v in pretrained_model.items():
            if k in model_dict and model_dict[k].size() == v.size():
                pretrained_dict[k] = v
                print(k)
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
                

class HRCNet(nn.Module):
    def __init__(self):
        super(HRCNet, self).__init__()
        self.head = nn.Sequential(nn.Conv2d(3, 32, 5, stride=1, padding=2, dilation=1), nn.ReLU(True), nn.Conv2d(32, 32, 3, stride=2, padding=1, dilation=1), nn.ReLU(True),
        nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=1), nn.ReLU(True))
        self.c1_1 = nn.Sequential(nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=1), nn.ReLU(True),nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=1), nn.ReLU(True),
                                  nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=1), nn.ReLU(True))
        self.c1_2 = nn.Sequential(nn.Conv2d(32, 32, 3, stride=2, padding=1, dilation=1), nn.ReLU(True),nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=1), nn.ReLU(True),
                                  nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=1), nn.ReLU(True))
        self.c2_1 = nn.Sequential(nn.Conv2d(32, 32, 3, stride=2, padding=1, dilation=1), nn.ReLU(True))
        self.c2_2 = nn.Sequential(nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=1), nn.ReLU(True))
        self.c2_3 = nn.Sequential(nn.Conv2d(32, 32, 3, stride=2, padding=1, dilation=1), nn.ReLU(True))
        
        self.c3_1 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1, dilation=1), nn.ReLU(True))
        self.c3_2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=1, padding=1, dilation=1), nn.ReLU(True))
        
        self.mid = make_layers([128]*2+[256]*4, in_channels=128, dilation=True, batch_norm=False, NL='relu')
        self.back1 = DualUp(256,256)
        self.back2 = DualUp(256,128)
        self.back3_1 = nn.Sequential(nn.Conv2d(128, 32, 3, stride=1, padding=1, dilation=1), nn.ReLU(True))
        self.back3_2 = nn.Sequential(nn.Conv2d(128, 32, 3, stride=1, padding=1, dilation=1), nn.ReLU(True))
        self.output_layer = nn.Sequential(nn.Conv2d(64,1,1), nn.ReLU(True))
        self.init_weights()
    def forward(self,x_in):
        x = self.head(x_in)#1/2
        x1_1 = self.c1_1(x)#1/2
        x1_2 = self.c1_2(x)#1/4
        x2_1 = torch.cat([self.c2_1(x1_1), self.c2_2(x1_2)],1)#1/4
        x2_2 = self.c2_3(x1_2)#1/8
        
        x3 = torch.cat([self.c3_1(x2_1), self.c3_2(x2_2)],1)#1/8
        x_mid = self.mid(x3)
        x_up_1 = self.back1(x_mid)#1/4 size
        x_up_2 = self.back2(x_up_1)#1/2 size
        x_up_3_1 = F.interpolate(x_up_2, size=x_in.shape[2:])
        x_up_3_1 = self.back3_1(x_up_3_1)
        x_up_3_2 = self.back3_2(x_up_2)
        x_up_3_2 = F.interpolate(x_up_3_2, size=x_in.shape[2:])
        x_up_3 = torch.cat([x_up_3_1, x_up_3_2],1)#original size
        output = self.output_layer(x_up_3) 
        return output
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):    
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #print('nn.Conv2d')
                print(m)
                #nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Module):
                pass
                #print('nn.Module')
                #print(m)
            else:
                print( 'else' )

if __name__ =='__main__':
    print("HRCNet Summary")
    net = HRCNet()
    from torchsummary import summary
    summary(net.cuda(), (3, 224, 224))