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


def get_patches(img_path):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    img_patch = []
    target_patch = []
    crop_size = (img.size[0]//2, img.size[1]//2)
    for dx, dy in (([0,0],[0,1],[1,0],[1,1])):
        img_p = img.crop((dx*crop_size[0], dy*crop_size[1], crop_size[0]*(1+dx), crop_size[1]*(1+dy)))
        target_p = target[crop_size[1]*dy:crop_size[1]*(1+dy), crop_size[0]*dx:crop_size[0]*(1+dx)]
        img_patch.append(img_p)
        target_patch.append(target_p)
    
    return img_patch, target_patch


def get_amp_gt_by_value(target, threshold=1e-5):
    seg_map = (target>threshold).astype(np.float32)
    return seg_map
    

def get_amp_gt_by_window(img, img_path):
    w,h = img.size
    # build ground truth attention map by value
    '''
    
    '''
    # build ground truth attention map by fixed window 25*25
    seg_map = np.zeros((h,w))
    if 'sh' in img_path:
        if 'A' in img_path:
            gt_dir = '/home/datamining/Datasets/CrowdCounting/shanghaitech/part_A_final/'
        else:
            gt_dir = '/home/datamining/Datasets/CrowdCounting/shanghaitech/part_B_final/'
        if 'rain' in img_path:
            gt_dir += 'train_data/ground_truth/'
        else:
            gt_dir += 'test_data/ground_truth/'
        gt_path = gt_dir + 'GT_IMG_' + img_path.split('IMG_')[-1][:-4] + '.mat'
    elif 'qnrf' in img_path:
        gt_dir = '/home/datamining/Datasets/CrowdCounting/UCF-QNRF_ECCV18/'
        if 'rain' in img_path:
            gt_dir += 'Train/'
        else:
            gt_dir += 'Test/'
        gt_path = gt_dir + 'img_'+ img_path.split('img_')[-1][:-4] + '_ann.mat'
    if 'qnrf' in img_path:
        points = io.loadmat(gt_path.replace('_rsz',''))["annPoints"]
    else:
        points = io.loadmat(gt_path)["image_info"][0, 0][0, 0][0].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= w) * (points[:, 1] >= 0) * (points[:, 1] <= h)
    points = points[idx_mask]
    for point in points:
        left = int(max(point[0]-12, 0))
        right = int(min(point[0]+12, w-1))
        up = int(max(point[1]-12, 0))
        down = int(min(point[1]+12, h-1))
        seg_map[up:down, left:right] = 1.0
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


def get_multi_random_patch(img_path, c_size, transform, n):
      batch = []
      for _ in range(n):
          img, t, c, amp = get_random_patch(img_path, c_size)
          if transform:
              img = transform(img)
          batch.append((img, torch.from_numpy(np.array(t)), torch.from_numpy(np.array(c)), torch.from_numpy(np.array(amp)) ))
      
      return batch


def get_random_patch(img_path, c_size):
    '''
    c_size:(cw,ch)
    '''
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(img_path.replace('images','ground_truth').replace('.jpg','.h5'), 'r')
    target = np.asarray(gt_file['density'])
    amp_gt = get_amp_gt_by_window(img, img_path)
    
    if random.random() > 0.5:
        target = np.fliplr(target)
        # to avoid "ValueError: some of the strides of a given numpy are negtive", since negtive strides might be used somewhere 
        target = target.copy()
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w,h = img.size[0], img.size[1]
    if c_size[0]<1 and c_size[1]<1:
        # crop by ratio
        c_size = [int(c_size[0]*w), int(c_size[1]*h)]
    c_size = [int(c_size[0]), int(c_size[1])]
    w_start = random.randint(0,w-c_size[0])
    h_start = random.randint(0,h-c_size[1])
    
    img_p = F.crop(img,h_start,w_start,c_size[1],c_size[0])
    target_p2 = target[h_start:h_start+c_size[1],w_start:w_start+c_size[0]]
    amp_gt_p = amp_gt[h_start:h_start+c_size[1],w_start:w_start+c_size[0]]
    '''
    gd_path = img_path.replace('IMG_','GT_IMG_').replace('images','ground_truth').replace('.jpg','.npy')
    keypoints = np.load(gd_path)
    # [x,y,distance] for each row, find the point in this sub-region
    idx_mask = (keypoints[:,0]>w_start)*(keypoints[:,0]<w_start+c_size[0])*(keypoints[:,1]>h_start)*(keypoints[:,1]<h_start+c_size[1])
    points = keypoints[idx_mask]
    print(len(points))
    target_p = np.zeros((c_size[1], c_size[0]))
    for p in points:
        # generate dmp of this image patch
        x,y = p[0]-w_start,p[1]-h_start
        pt2d = np.zeros(target_p.shape, dtype=np.float32)
        pt2d[int(y),int(x)] = 1.0
        target_p += ndimage.filters.gaussian_filter(pt2d, p[2])
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.imshow(img.crop((w_start,h_start,w_start+c_size[0],h_start+c_size[1])))
    plt.subplot(2,2,2)
    plt.imshow(target_p, cmap=CM.jet)
    plt.subplot(2,2,3)
    gt_file = h5py.File(img_path.replace('images','ground_truth').replace('.jpg','.h5'), 'r')
    target = np.asarray(gt_file['density'])
    target_p2 = target[h_start:h_start+c_size[1],w_start:w_start+c_size[0]]
    plt.imshow(target_p2, cmap=CM.jet)
    plt.savefig('./figs/gen_sub_dmp.png')
    '''
    
    return img_p, target_p2, target_p2.sum(), amp_gt_p


def get_batch(root, start_index, size, n, m):
    img_patch = []
    target_patch = []
    for i in range(n):
        index = start_index+i
        img_path = root[index%len(root)]
        img = Image.open(img_path).convert('RGB')
        dm = h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth'), 'r')
        target = np.asarray(dm['density'])
        w,h = img.size[0:2]
        for j in range(m):
            # get size*size patch, x in [0, w-size], y in [0,h-size]
            x = random.randint(0,w-size)
            y = random.randint(0,h-size)
            img_p = img.crop((x,y,x+size,y+size))
            target_p = target[y:y+size, x:x+size]
            img_patch.append(img_p)
            target_patch.append(target_p)
    return img_patch, target_patch
        

def load_data(img_path, downsample_ratio, len_expansion, index, length, crop_scale, mode, curriculum_ratio, test):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    w,h = img.size
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    
    amp_gt = get_amp_gt_by_value(target, threshold=1e-5)
    #amp_gt = get_amp_gt_by_window(img, img_path)
    
    if crop_scale <= 1.0:
        crop_size = (int(img.size[0]*crop_scale), int(img.size[1]*crop_scale))
    else:
        crop_size = (int(crop_scale), int(crop_scale))
    if mode=='fixed+random': # 4 fixed and 5 random. set the seed by index to get fixed cropped patches. or just crop directly, which will lead to a changing dataset.
        if len_expansion * index < length:
            dx, dy = 0, 0
        elif len_expansion * index < 2 * length:
            dx, dy = 0, h-crop_size[1]
        elif len_expansion * index < 3 * length:
            dx, dy = w-crop_size[0], 0
        elif len_expansion * index < 4 * length:
            dx, dy = w-crop_size[0], h-crop_size[1]
        else:
            '''
            random.seed(index)
            dx = random.randint(0, w-crop_size[0])
            random.seed(index + 0.5)
            dy = random.randint(0, h-crop_size[1])
            '''
            dx = random.randint(0, w-crop_size[0])
            dy = random.randint(0, h-crop_size[1])
        
    elif mode=='random':
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
    elif mode=='mixed':
        m_scale = 0.3 + 0.1 * (len_expansion*index//length) #  cal the scale. [0,1/len_expansion] scale=0.3, [1/len_expansion,2/len_expansion] scale=0.4, etc
        
        crop_size = (int(img.size[0]*m_scale), int(img.size[1]*m_scale))
        dx = random.randint(0, w-crop_size[0])
        dy = random.randint(0, h-crop_size[1])
    elif mode=='curriculum':
        c_scale = 0.4 + 0.1 * (len_expansion * index // length) * curriculum_ratio
        if random.random()>0.99 and len_expansion * index // length == 2:
            print('len_expansion:{}, index:{}, length:{}, c_scale:{} '.format(len_expansion, index, length, c_scale))
        crop_size = (int(img.size[0]*c_scale), int(img.size[1]*c_scale))
        dx = random.randint(0, w-crop_size[0])
        dy = random.randint(0, h-crop_size[1])
    #crop by dx,dy
    if mode in ['fixed', 'random', 'fixed+random']:
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
    
    
    # gamma transform
    '''
    if random.random() > 0.5:
        image = img_as_float(img)
        # gamma_img: np.array(dtype=float64) ranging [0,1]
        if random.random() > 0.5:
            gamma_img = exposure.adjust_gamma(image, 1.5)
        else:
            gamma_img = exposure.adjust_gamma(image, 0.5)
        gamma_img = gamma_img * 255
        gamma_img = np.uint8(gamma_img)
        img = Image.fromarray(gamma_img)
        '''
    count = target.sum()
    
    #downsample the output if needed. must multiply a ratio**2 to ensure that the count over the density map is not changed.
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
        if mode=='fixed+random':
            len_expansion = 9
            root = root*len_expansion
        elif mode=='mixed':
            len_expansion = 5
            root = root*len_expansion #0.3,0.4,0.5,0.6,0.7
        elif mode=='curriculum':
            len_expansion = 3
            root = root*len_expansion #0.4 initially, then gradually changed to 0.4,0.5,0.6
        elif mode=='fixed':
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


class PatchesSet(Dataset):
    def __init__(self, root, transform):
        self.nsamples = len(root)
        self.root = root
        self.transform = transform
    def __getitem__(self, index):
        img_patch, target_patch = get_patches(self.root[index])
        if self.transform:
            for i in range(len(img_patch)):
                img_patch[i] = self.transform(img_patch[i])
        return img_patch, target_patch, count
    def __len__(self):
        return self.nsamples


class PatchSet(Dataset):
    def __init__(self, root, transform, c_size=(128,128),crop_n=4):
        '''
        c_size:crop w, crop h, if >1, crop by [cw,ch], if <1,crop by ratio:[imgw*c_size[0],imgh*c_size[1]]
        '''
        self.c_size = c_size
        self.root = root
        self.transform = transform
        self.crop_n = crop_n
    def __getitem__(self, index):
        return get_multi_random_patch(self.root[index], self.c_size, self.transform, self.crop_n)
        '''
        img_patch, target_patch, count = get_random_patch(self.root[index], c_size=self.c_size)
        if self.transform:
            img_patch = self.transform(img_patch)
        return img_patch, target_patch, count'''
    def __len__(self):
        return len(self.root)


class BatchSet(Dataset):
    def __init__(self, root, transform, size=(512,384), m=4, n=1):
        self.size=size if len(size)==2 else (size,size)
        print('BatchSet, size:{}'.format(self.size))
        self.root = root
        self.transform = transform
        self.m = m
        self.n = n
    def __getitem__(self, index):
        # get n*m patches, n images, m size*size patches for each image.
        img_path = self.root[index]
        img = Image.open(img_path).convert('RGB')
        dm = h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth'), 'r')
        target = np.asarray(dm['density'])
        w,h = img.size[0:2]
        x = random.randint(0,w-self.size[0])
        y = random.randint(0,h-self.size[1])
        img_p = img.crop((x,y,x+self.size[0],y+self.size[1]))
        target_p = target[y:y+self.size[1], x:x+self.size[0]]
        if self.transform:
            img_p = self.transform(img_p)
        return img_p, target_p
        '''
        img_patch, target_patch = get_batch(self.root, index, self.size, self.n, self.m)
        if self.transform:
            for i in range(len(img_patch)):
                img_patch[i] = self.transform(img_patch[i])
        return img_patch, target_patch
        '''
    def __len__(self):
        return len(self.root)


if __name__ == '__main__':
    pass
    #get_random_patch('/home/datamining/Datasets/CrowdCounting/shanghaitech/part_A_final/train_data/images/IMG_195.jpg', (500,500))