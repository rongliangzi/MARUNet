from torch.utils.data import Dataset
from PIL import Image
import h5py
import numpy as np
import cv2
import random
import torchvision.transforms.functional as F
from skimage import exposure, img_as_float
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import torch
from matplotlib import cm as CM
import collections
import scipy.io as io


def get_amp_gt_by_value(target, threshold=1e-5):
    seg_map = (target>threshold).astype(np.float32)
    return seg_map
    
    
def my_collate_fn(batch):
    #print(type(batch[0][1]))
    if torch.is_tensor(batch[0]):
        print('istensor')
    elif isinstance(batch[0], collections.Sequence):
        out = None
        new_batch = []
        for b in batch:
            #print(len(b))
            for img_t_c_amp_tuple in b:
                new_batch.append(img_t_c_amp_tuple)
        #print(len(new_batch))
        transposed = list(zip(*new_batch))
        img = transposed[0]
        img = torch.stack(img,0)
        t = transposed[1]
        t = torch.stack(t,0)
        c = transposed[2]
        c = torch.stack(c,0)
        amp = transposed[3]
        amp = torch.stack(amp, 0)
        return [img, t, c, amp]
    

def bayes_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    if len(transposed_batch)==5:
        amps = torch.stack(transposed_batch[4], 0)
        return images, points, targets, st_sizes, amps
    else:
        return images, points, targets, st_sizes


def load_data(img_path, downsample_ratio, len_expansion, index, length, crop_scale, mode, curriculum_ratio, test):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    w,h = img.size
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    
    amp_gt = get_amp_gt_by_value(target, threshold=1e-5)
    
    if crop_scale <= 1.0:
        crop_size = (int(img.size[0]*crop_scale), int(img.size[1]*crop_scale))
    else:
        crop_size = (int(crop_scale), int(crop_scale))
    if mode=='random':
        dx = random.randint(0, w-crop_size[0])
        dy = random.randint(0, h-crop_size[1])
        
    elif mode=='fixed':
        # len_expansion should be set to 4
        if len_expansion * index < length:
            dx, dy = 0, 0
        elif len_expansion * index < 2 * length:
            dx, dy = 0, h-crop_size[1]
        elif len_expansion * index < 3 * length:
            dx, dy = w-crop_size[0], 0
        elif len_expansion * index < 4 * length:
            dx, dy = w-crop_size[0], h-crop_size[1]
    #crop by dx,dy
    if mode in ['fixed', 'random']:
        if crop_size[0] < img.size[0]:
            img = img.crop((dx, dy, crop_size[0]+dx, crop_size[1]+dy))
            target = target[dy:crop_size[1]+dy, dx:crop_size[0]+dx]
            amp_gt = amp_gt[dy:crop_size[1]+dy, dx:crop_size[0]+dx]
        else:
            print('crop size > img size in load_data()!')
    # if mode =='one', do not do crop
    
    # flip left-right at p=0.5
    if not test and random.random() > 0.5:
        target = np.fliplr(target)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    count = target.sum()
    
    # downsample the output if needed. must multiply a ratio**2 to ensure that the count over the density map is not changed.
    if downsample_ratio > 1:
        target = cv2.resize(target, (int(target.shape[1]/downsample_ratio),int(target.shape[0]/downsample_ratio)), interpolation=cv2.INTER_CUBIC) * (downsample_ratio**2)
        amp_gt = cv2.resize(amp_gt, (target.shape[1], target.shape[0]))
        amp_gt = (amp_gt>0.5).astype(np.float32)
    else:
        # to avoid "ValueError: some of the strides of a given numpy are negtive", since negtive strides might be used somewhere 
        target = target.copy()
    
    return img, target, count, amp_gt


class CrowdDataset(Dataset):
    def __init__(self, root, transform, mode, downsample_ratio=8, crop_scale=0.5, curriculum_ratio=0.0, test=False): # curriculum_ratio only work when mode is 'curriculum'
        if mode=='fixed':
            len_expansion = 4
            root = root*len_expansion
        else:
            len_expansion = 1
        
        self.nsamples = len(root)
        self.len_expansion = len_expansion
        self.crop_scale = crop_scale
        self.root = root
        self.downsample_ratio = downsample_ratio
        self.transform = transform
        self.mode = mode
        self.curriculum_ratio = curriculum_ratio
        self.test = test
    def __getitem__(self, index):
        
        img, target, count, amp = load_data(self.root[index], self.downsample_ratio, self.len_expansion, index, self.nsamples, self.crop_scale, self.mode, self.curriculum_ratio, self.test)
        if self.transform:
            img = self.transform(img)
        return img, target, count, amp
    def __len__(self):
        return self.nsamples


if __name__ == '__main__':
    pass
    #get_random_patch('/home/datamining/Datasets/CrowdCounting/shanghaitech/part_A_final/train_data/images/IMG_195.jpg', (500,500))