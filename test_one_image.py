import cv2
import numpy as np
import torch
from torch.autograd import Variable
from modeling import *
import os
from matplotlib import cm as CM
import matplotlib.pyplot as plt
# set your pretrained model path here
model_paths = {'sha':{'MARNet':"/home/datamining/Models/CrowdCounting/MARNet_d1_sha_random_cr0.5_3avg-ms-ssim_v50_amp0.11_bg_lsn6.pth",
                              'U_VGG':"/home/datamining/Models/CrowdCounting/U_VGG_d1_sha_random_cr0.5_ap3avg-ms-ssim_v50_bg_lsn6.pth",
                              'CSRNet':"/home/datamining/Models/CrowdCounting/PartAmodel_best.pth.tar"},
                       'shb':{'MARNet':"/home/datamining/Models/CrowdCounting/MARNet_d1_shb_random_cr0.5_3avg-ms-ssim_v50_amp0.15_bg_lsn6.pth",
                              'U_VGG':"/home/datamining/Models/CrowdCounting/U_VGG_d1_shb_random_cr0.5_3avg-ms-ssim_v50_bg_lsn6.pth",
                              'CSRNet':"/home/datamining/Models/CrowdCounting/partBmodel_best.pth.tar"},
                       'qnrf':{'MARNet':"/home/datamining/Models/CrowdCounting/MARNet_d1_qnrf_random_cr0.5_3avg-ms-ssim_v50_amp0.16_bg_lsn6.pth"},
                               'U_VGG':"/home/datamining/Models/CrowdCounting/U_VGG_d1_qnrf_random_cr0.5_3avg-ms-ssim_v50_bg_lsn6.pth",
                               }

def preprocess_image(cv2im):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def load_model(models, model_paths, dataset='sha'):
    pretrained_models = {}
    for model in models:
        if model == 'MARNet':
            pretrained_model = MARNet(load_model=model_paths[dataset][model], downsample=1, objective='dmp+amp')
        elif model == 'MSUNet':
            pretrained_model = U_VGG(load_model=model_paths[dataset]['U_VGG'], downsample=1)
        elif model == 'CSRNet':
            pretrained_model = CSRNet(load_model=model_paths[dataset][model], downsample=8)
        pretrained_models[model]=pretrained_model
    return pretrained_models
    
    
def img_test(pretrained_model, img_path="/home/datamining/Pictures/IMG_98.jpg", divide=50, ds=1):
    img = cv2.imread(img_path)
    img = preprocess_image(img)
    if torch.cuda.is_available():
        img = img.cuda()
    outputs = pretrained_model(img)
    if torch.cuda.is_available():
        dmp = outputs[0].squeeze().detach().cpu().numpy()
        amp = outputs[-1].squeeze().detach().cpu().numpy()
    else:
        dmp = outputs[0].squeeze().detach().numpy()
        amp = outputs[-1].squeeze().detach().numpy()
    dmp = dmp/divide
    print('estimated people count: ', dmp.sum())
    return dmp

if __name__ == '__main__':
    model = load_model(['MARNet'],None)
    if torch.cuda.is_available():
        model = model.cuda()
    img_path = "/home/datamining/Pictures/IMG_10.jpg"
    dmp = img_test(model['MARNet'], img_path, divide=50, ds=1)
