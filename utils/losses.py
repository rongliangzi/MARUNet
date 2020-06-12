import torch.nn as nn
from .pytorch_ssim import *


class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(64, 64, 3), nn.Conv2d(64, 128, 3), nn.PReLU(), nn.Conv2d(128, 256, 3), nn.PReLU(), nn.Conv2d(128, 256, 3), nn.Conv2d(128, 256, 3), nn.PReLU(), nn.Conv2d(256, 1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(x):
        return torch.abs(self.conv(x))
    
    
def cal_lc_loss(output, target, sizes=(1,2,4)):
    '''
    Multi-scale density level consistency loss proposed by "Dense Scale Network for Crowd Counting"
    '''
    criterion_L1 = nn.L1Loss()
    Lc_loss = None
    for s in sizes:
        pool = nn.AdaptiveAvgPool2d(s)
        est = pool(output)
        gt = pool(target)
        if Lc_loss:
            Lc_loss += criterion_L1(est, gt)
        else:
            Lc_loss = criterion_L1(est, gt)
    return Lc_loss


def cal_spatial_abstraction_loss(output, target, levels=3):
    '''
    Spatial Abstraction Loss. proposed by "Crowd Counting and Density Estimation by Trellis Encoder-Decoder Networks" CVPR2019
    '''
    criterion = nn.MSELoss()
    sa_loss = None
    est = output
    gt = target
    pool = nn.MaxPool2d(kernel_size=2,stride=2)
    for _ in range(levels):
        
        est = pool(est)
        gt = pool(gt)
        if sa_loss:
            sa_loss += criterion(est, gt)
        else:
            sa_loss = criterion(est, gt)
    return sa_loss


def cal_spatial_correlation_loss(output, target):
    '''
    Spatial Correlation Loss. proposed by "Crowd Counting and Density Estimation by Trellis Encoder-Decoder Networks" CVPR 2019
    '''
    sc_loss = 1.0 - ((output * target).sum()+1e-6) / (((output**2).sum() * (target**2).sum())**0.5+1e-6)
    return sc_loss


def _ssim(img1, img2, window, window_size, channel, size_average = True, dilation=1, full=False):
    kernel_size = window_size + (dilation - 1) * (window_size - 1) - 1
    mu1 = F.conv2d(img1, window, padding = kernel_size//2, dilation = dilation, groups = channel)
    mu2 = F.conv2d(img2, window, padding = kernel_size//2, dilation = dilation, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = kernel_size//2, dilation = dilation, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = kernel_size//2, dilation = dilation, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = kernel_size//2, dilation = dilation, groups = channel) - mu1_mu2

    C1 = (0.01*1)**2
    C2 = (0.03*1)**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)
    
    if size_average:
        if full:
            return ssim_map.mean(), cs
        else:
            return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
        

def cal_avg_ms_ssim(img1, img2, level=5, weights=[1]*5, window_size=11):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if len(weights)!=level:
        weights = [1] * level
    weights = torch.FloatTensor(weights)
    weights = weights/weights.sum()
    mssim = []
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
        weights = weights.cuda(img1.get_device())
    window = window.type_as(img1)
    for i in range(level):
        ssim_value = _ssim(img1, img2, window, window_size, channel, True, 1, False)
        if i==0:
            avg_loss = weights[i] * (1.0 - ssim_value)
        else:
            avg_loss += weights[i] * (1.0 - ssim_value)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))
    
    return avg_loss
    
    
def cal_ms_ssim(img1, img2, level=5, window_size=5, dilations=[1,1,1,1,1]):
    img1 = (img1 + 1e-8) / (img1.max() + 1e-8)
    img2 = (img2 + 1e-8) / (img2.max() + 1e-8)
    img1 = img1 * 255.0
    img2 = img2 * 255.0
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    mssim = []
    cs = []
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
        weights = weights.cuda(img1.get_device())
    window = window.type_as(img1)
    for i in range(level):
        kernel_size = window_size + (dilations[i] - 1) * (window_size - 1) - 1
        mu1 = F.conv2d(img1, Variable(window, requires_grad=False), padding = kernel_size//2, dilation=dilations[i])
        mu2 = F.conv2d(img2, Variable(window, requires_grad=False), padding = kernel_size//2, dilation=dilations[i])
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        
        sigma1_sq = F.conv2d(img1*img1, Variable(window, requires_grad=False), padding = kernel_size//2, dilation=dilations[i]) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, Variable(window, requires_grad=False), padding = kernel_size//2, dilation=dilations[i]) - mu2_sq
        sigma12 = F.conv2d(img1*img2, Variable(window, requires_grad=False), padding = kernel_size//2, dilation=dilations[i]) - mu1_mu2
        
        C1 = (0.01*255)**2
        C2 = (0.03*255)**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        ssim_value = ssim_map.mean()
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs_value = torch.mean(v1 / v2)
        
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))
        mssim.append(ssim_value)
        cs.append(cs_value)
    mssim = torch.stack(mssim, dim=0)
    cs = torch.stack(cs, dim=0)
    weights = Variable(weights, requires_grad=False)
    #ms_ssim = torch.prod(cs[:-1]**weights[:-1].unsqueeze(1) * mssim[-1]**weights[-1], dim=0)
    ms_ssim = torch.prod(mssim**weights)
    return 1-ms_ssim
    
    
def cal_dms_ssim_loss(img1, img2, level=5, window_size=5, dilations=[1,2,3,6,9]):
    
    return cal_ms_ssim(img1, img2, level, window_size, dilations)