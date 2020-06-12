'''MobileNetV2 as backbone
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileCountingNet(nn.Module):
    def __init__(self, bn=True, width_mult=1.0, round_nearest=8):
        super(MobileCountingNet, self).__init__()
        self.bn = bn
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        # (expansion, out_planes, num_blocks, stride)
        # [6, 64, 4, 2] -> [6, 64, 4, 1]
        # [6, 160, 3, 2] -> [6, 160, 3, 1]
        inverted_residual_setting = [[1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 1],
                [6, 96, 3, 1],
                [6, 160, 3, 1],
                [6, 320, 1, 1],
                ]
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))
                             
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.bk1 = ConvBNReLU(self.last_channel, 128, 3)
        self.bk2 = ConvBNReLU(128, 128, 3)
        self.output_layers = nn.Conv2d(128, 1, 1)
        self._init_weights()
        
    def forward(self, x_in):
        
        x = self.features(x_in)
        x = F.interpolate(x, scale_factor=2)#w//4
        x = self.bk1(x)
        x = F.interpolate(x, scale_factor=2)#w//2
        x = self.bk2(x)
        x = F.interpolate(x, size=x_in.shape[2:])
        out = self.output_layers(x)
        return out
    
    def _init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        pretrained_dict = dict()
        model_dict = self.state_dict()
        pretrained_model = torch.load("/home/datamining/Models/mobilenet_v2-b0353104.pth")
        # load the pretrained model parameters
        for k, v in pretrained_model.items():
            if k in model_dict and model_dict[k].size() == v.size():
                pretrained_dict[k] = v
                print(k)
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        

def test():
    net = MobileCountingNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()
