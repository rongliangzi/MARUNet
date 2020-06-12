import matplotlib as mpl
# we cannot use remote server's GUI, so set this  
mpl.use('Agg')
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
from modeling import *
import os
import matplotlib.pyplot as plt
from matplotlib import cm as CM


def preprocess_image(cv2im, resize_im=False):
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
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
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
 
 
class FeatureVisualization():
    def __init__(self,img_path,selected_layer):
        self.img_path = img_path
        self.selected_layer = selected_layer
        self.pretrained_model = MARNet(load_model="/home/datamining/Models/CrowdCounting/MARNet_d1_sha_random_cr0.5_3avg-ms-ssim_v50_amp0.11_bg_lsn6.pth", downsample=1, objective='dmp+amp', save_feature=True)
 
    def process_image(self):
        img = cv2.imread(self.img_path)
        img = preprocess_image(img)
        self.img = img
        return img
        
    def get_single_feature(self, num):
        input_img = self.process_image()
        print('input_img.shape:', input_img.shape)
        x=input_img
        outputs = self.pretrained_model(x)
        amp4, amp3, amp2, amp1, amp0 = outputs[-5:]
        fea_d = dict()
        amp_d = dict()
        amp_d['amp4'] = amp4
        amp_d['amp3'] = amp3
        amp_d['amp2'] = amp2
        amp_d['amp1'] = amp1
        amp_d['amp0'] = amp0
        
        fea_d['xb4_before'] = self.pretrained_model.xb4_before[:,0:num,:,:].squeeze_()
        fea_d['xb4_after'] = self.pretrained_model.xb4_after[:,0:num,:,:].squeeze_()
        
        fea_d['xb3_before'] = self.pretrained_model.xb3_before[:,0:num,:,:].squeeze_()
        fea_d['xb3_after'] = self.pretrained_model.xb3_after[:,0:num,:,:].squeeze_()
        
        fea_d['xb2_before'] = self.pretrained_model.xb2_before[:,0:num,:,:].squeeze_()
        fea_d['xb2_after'] = self.pretrained_model.xb2_after[:,0:num,:,:].squeeze_()
        
        fea_d['xb1_before'] = self.pretrained_model.xb1_before[:,0:num,:,:].squeeze_()
        fea_d['xb1_after'] = self.pretrained_model.xb1_after[:,0:num,:,:].squeeze_()
        
        fea_d['xb0_before'] = self.pretrained_model.xb0_before[:,0:num,:,:].squeeze_()
        fea_d['xb0_after'] = self.pretrained_model.xb0_after[:,0:num,:,:].squeeze_()
        
        return amp_d, fea_d
 
    def save_feature_to_img(self):
        #to numpy
        num = 4
        amps, features = self.get_single_feature(num)
        height, width = self.img.shape[2:]
        for item in amps.items():
            k,v = item
            v = v.squeeze_().data.numpy()
            fig, ax = plt.subplots()
            ax.imshow(v, cmap=CM.jet)
            fig.set_size_inches(width/400.0, height/400.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.axis('off')
            plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
            plt.margins(0,0)
            plt.savefig('figs/visual_feature/sha130_{}.jpg'.format(k), dpi=300)
            plt.clf()
            
        for item in features.items():
            k, v = item
            
            col = 2
            row = num // col
            
            plt.figure(figsize=(width*col/400.0,height*row/400.0))
            for i in range(row):
                for j in range(col):
                    feature = v[i * col + j].data.numpy()
                    plt.subplot(row, col, i * col + j + 1)
                    plt.imshow(feature, cmap=CM.jet)
                    plt.axis('off')
            plt.subplots_adjust(top=0.99,bottom=0.01,left=0.01,right=0.99,hspace=0.01,wspace=0.01)
            plt.margins(0,0)
            
            plt.savefig('figs/visual_feature/sha130_{}.jpg'.format(k), dpi=300)
            plt.clf()


if __name__=='__main__':
    # get class
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    myClass=FeatureVisualization("/home/datamining/Datasets/CrowdCounting/shanghaitech/part_A_final/test_data/images/IMG_130.jpg",5)
    #print (myClass.pretrained_model)
 
    myClass.save_feature_to_img()
 