from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import h5py
import cv2


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

# cal overlap area of each head
def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area


class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train'):

        self.root_path = root_path
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.method = method

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        
        if self.method == 'train':
            if 'qnrf' in img_path or 'UCF-Train-Val-Test' in img_path:
                img_name = img_path.split('/')[-1].split('.')[-2]
                h5_path = "/home/datamining/Datasets/CrowdCounting/qnrf_.1024_a/"
                if 'rain' in img_path:
                    h5_path += 'train/'
                else:
                    h5_path += 'test/'
            elif 'sha' in img_path:
                img_name = img_path.split('/')[-1].split('.')[-2]
                h5_path = "/home/datamining/Datasets/CrowdCounting/shanghaitech/part_A_final/"
                if 'train' in img_path:
                    h5_path += 'train_data/ground_truth/'
                else:
                    h5_path += 'test_data/ground_truth/'
            h5_path = h5_path + img_name + '.h5'
            h5gt = h5py.File(h5_path, 'r')
            h5gt = np.asarray(h5gt['density'])
            if img.size[0]!=h5gt.shape[1] or img.size[1]!=h5gt.shape[0]:
                h5gt = cv2.resize(h5gt, img.size)#w,h
            amp_gt = (h5gt>1e-5).astype(np.float32)
            keypoints = np.load(gd_path)
            img, kpts, target, st_size = self.train_transform(img, keypoints)
            return (img, kpts, target, st_size, torch.from_numpy(amp_gt))
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        st_size = min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) > 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        # constrain the adaptive distance in [4.0,128.0]
        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)
        # each cood minus 0.5*adaptive distance
        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        inner_area = cal_innner_area(j, i, j+w, i+h, bbox)
        origin_area = nearest_dis * nearest_dis
        # each row in inner_area: the area in crop
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        # only use heads that overlap is higher than 30%
        mask = (ratio >= 0.3)

        target = ratio[mask]
        keypoints = keypoints[mask]
        #keypoints = keypoints[:, :2] - [j, i]  # change coodinate
        keypoints[:, :2] -= [j, i]
        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(target.copy()).float(), st_size
