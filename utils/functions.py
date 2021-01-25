import torch
import logging
import torch.nn as nn
import numpy as np
import math
from .pytorch_ssim import *
import torch.nn.functional as F
import torchvision.transforms as transforms
import glob
import os
from dataset import *
from dataset import CrowdDataset
import json
from datasets.crowd import Crowd


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class Discriminator(nn.Module):
    '''
    Discriminator to optimize Adversarial Loss in "Residual Regression with Semantic Prior for Crowd Counting"
    '''
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPooling(2,2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.MaxPooling(2,2),
            nn.Conv2d(256, 1, 3),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)

        return validity
        
        
def get_loader(train_path, test_path, args):
    train_img_paths = []
    for img_path in glob.glob(os.path.join(train_path, '*.jpg')):
        train_img_paths.append(img_path)
    test_img_paths = []
    for img_path in glob.glob(os.path.join(test_path, '*.jpg')):
        test_img_paths.append(img_path)
    
    if 'bayes' in args.loss:
        bayes_dataset = Crowd(train_path, args.crop_scale, args.downsample, False, 'train')
        train_loader = torch.utils.data.DataLoader(bayes_dataset, collate_fn=bayes_collate, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(Crowd(test_path, args.crop_scale, args.downsample, False, 'val'),batch_size=1, num_workers=8, pin_memory=True)
    elif args.bn > 0:
        bn_dataset=PatchSet(train_img_paths, transform, c_size=(args.crop_scale,args.crop_scale), crop_n=args.random_crop_n)
        train_loader = torch.utils.data.DataLoader(bn_dataset, collate_fn=my_collate_fn, shuffle=True, batch_size=args.bs, num_workers=8, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(CrowdDataset(test_img_paths, transform, mode='one', downsample_ratio=args.downsample, test=True), shuffle=False, batch_size=1, pin_memory=True)
    else:
        single_dataset = CrowdDataset(train_img_paths, transform, args.crop_mode, args.downsample, args.crop_scale)
        train_loader = torch.utils.data.DataLoader(single_dataset, shuffle=True, batch_size=1, num_workers=8, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(CrowdDataset(test_img_paths, transform, mode='one', downsample_ratio=args.downsample, test=True), shuffle=False, batch_size=1, pin_memory=True)
    
    return train_loader, test_loader, train_img_paths, test_img_paths
    

def get_loader_json(args):
    with open(args.val_json, 'r') as outfile:       
        val_list = json.load(outfile)
    val_loader = torch.utils.data.DataLoader(CrowdDataset(val_list, transform, mode='one', downsample_ratio=args.downsample, test=True), shuffle=False, batch_size=1, pin_memory=True)
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    train_loader = torch.utils.data.DataLoader(CrowdDataset(train_list, transform, args.crop_mode, args.downsample, args.crop_scale), shuffle=True, batch_size=1, num_workers=8, pin_memory=True)
    return train_loader, val_loader
    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count


def linear_warm_up_lr(optimizer, epoch, warm_up_steps, lr):
    for param_group in optimizer.param_groups:
        warm_lr = lr*(epoch+1.0)/warm_up_steps
        param_group['lr'] = warm_lr


def get_logger(filename):
    logger = logging.getLogger('train_logger')

    while logger.handlers:
        logger.handlers.pop()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, 'w')
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s], ## %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def val(model, test_loader, factor=1.0, verbose=False, downsample = 8, roi=None):
    print('validation on whole images!')
    model.eval()
    mae, rmse = 0.0, 0.0
    psnr = 0.0
    ssim = 0.0
    with torch.no_grad():
        for it,data in enumerate(test_loader):
            img, target, count = data[0:3]
            
            img = img.cuda()
            target = target.unsqueeze(1).float().cuda()
            output = model(img)
            if isinstance(output, tuple):
                if len(output)==2:
                    dmp, amp = output
                    #hard_amp = (amp > 0.5).float()
                    dmp = dmp * amp
                else:# more than 2 outputs
                    dmp = output[0]
            else:
                dmp = output
            if roi:
                roi = cv2.resize(roi, (dmp.shape[3], dmp.shape[2]))
                roi = torch.from_numpy(roi).unsqueeze(0).unsqueeze(0)
                dmp = dmp * roi
            dmp = dmp/factor
            est_count = dmp.sum().item()
            mae += abs(est_count - count.item())
            rmse += (est_count - count.item())**2
            if verbose:
                divide = max(dmp.max(), target.max())
                dmp_n = dmp / divide
                target_n = target / divide
                mse = torch.mean((dmp_n - target_n)**2).float()
                
                psnr_t = 10 * math.log(1/mse, 10)
                psnr += psnr_t
                
                ssim_t = cal_ssim(dmp_n, target_n)
                ssim += ssim_t
                print('gt:{:.1f}, est:{:.1f}, ssim:{:.2f}, psnr:{:.2f}'.format(count.item(),est_count,ssim_t,psnr_t))
            elif it < 10:
                print('gt:{:.1f}, est:{:.1f}'.format(count.item(),est_count))
            
    mae /= len(test_loader)
    rmse /= len(test_loader)
    rmse = rmse**0.5
    psnr /= len(test_loader)
    ssim /= len(test_loader)
    if verbose:
        print('psnr:{:.2f}, ssim:{:.2f}'.format(psnr, ssim))
    return mae, rmse
    

def test_ssim():
    img1 = (torch.rand(1, 1, 16, 16))
    img2 = (torch.rand(1, 1, 16, 16))

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()
    print(torch.max(img1))
    print(torch.max(img2))
    print(max(torch.max(img1),torch.max(img2)))
    print(cal_ssim(img1, img2).float())
    
    
    
# validate on bayes dataloader
def val_bayes(model, test_loader, factor=1.0, verbose=False):
    print('validation on bayes loader!')
    model.eval()
    epoch_res=[]
    for it,(inputs, count, name) in enumerate(test_loader):
        inputs = inputs.cuda()
        # inputs are images with different sizes
        assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        with torch.set_grad_enabled(False):
            output = model(inputs)
            if isinstance(output, tuple):
                if len(output)==2:
                    dmp, amp = output
                    dmp = dmp * amp
                else:# more than 2 outputs
                    dmp = output[0]
            else:
                dmp = output
            
            est = torch.sum(dmp).item()
            res = count[0].item() - est
            if verbose:
                print('gt:{:.1f}, est:{:.1f}'.format(count[0].item(),est))
            elif it<10:
                print('gt:{:.1f}, est:{:.1f}'.format(count[0].item(),est))
            epoch_res.append(res)
    epoch_res = np.array(epoch_res)
    rmse = np.sqrt(np.mean(np.square(epoch_res)))
    mae = np.mean(np.abs(epoch_res))
    return mae, rmse


# validate with 4 non-overlapping patches
def val_patch(model, test_loader, factor=1.0, verbose=False):
    print('validaiton on 4 quarters!')
    model.eval()
    mae, rmse = 0.0, 0.0
    with torch.no_grad():
        for it, data in enumerate(test_loader):
            img, _, count = data[0:3]
            h,w = img.shape[2:]
            h_d = h//2
            w_d = w//2
            
            img_1 = (img[:,:,:h_d,:w_d].cuda())
            img_2 = (img[:,:,:h_d,w_d:].cuda())
            img_3 = (img[:,:,h_d:,:w_d].cuda())
            img_4 = (img[:,:,h_d:,w_d:].cuda())
            img_patches = [img_1, img_2, img_3, img_4]
            est_count = 0
            for img_p in img_patches:
                output = model(img_p)
                if isinstance(output, tuple):
                    if len(output)==2:
                        dmp, amp = output
                        #hard_amp = (amp > 0.5).float()
                        dmp = dmp * amp
                    else:# more than 2 outputs
                        dmp = output[0]
                else:
                    dmp = output
                est_count += dmp.sum().item()/factor
            if verbose:
                print('gt:{:.1f}, est:{:.1f}'.format(count.item(),est_count))
            elif it < 10:
                print('gt:{:.1f}, est:{:.1f}'.format(count.item(),est_count))
            mae += abs(est_count - count.item())
            rmse += (est_count - count.item())**2
    mae /= len(test_loader)
    rmse /= len(test_loader)
    rmse = rmse**0.5
    return mae, rmse
