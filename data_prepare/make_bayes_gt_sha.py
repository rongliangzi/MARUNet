from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse

# ensure that the shorter side length in [min_size, max_size], if not, resize. return the h,w,ratio
def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    square = np.sum(point*point, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    # cal the avg distance of 3 nearest head 
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('.jpg', '.mat').replace('images','ground_truth').replace('IMG','GT_IMG')
    
    points = loadmat(mat_path)["image_info"][0, 0][0, 0][0].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        # resize, and each coordinate of head also multiply the ratio.
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default='/home/datamining/Datasets/CrowdCounting/shanghaitech/part_B_final/',
                        help='original data directory')
    parser.add_argument('--data-dir', default='/home/datamining/Datasets/CrowdCounting/shb_bayes_512/',
                        help='processed data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    min_size = 512
    max_size = 2048
    
    for phase in ['train_data', 'test_data']:
        sub_dir = os.path.join(args.origin_dir, phase+'/images/')
        print(sub_dir)
        if 'rain' in phase:
            sub_phase_list = ['train']
            for sub_phase in sub_phase_list:
                sub_save_dir = os.path.join(save_dir, sub_phase)
                if not os.path.exists(sub_save_dir):
                    os.makedirs(sub_save_dir)
                for im_path in glob(os.path.join(sub_dir, '*.jpg')):
                    name = os.path.basename(im_path)
                    print(name)
                    im, points = generate_data(im_path)
                    if sub_phase == 'train':
                        dis = find_dis(points)
                        # each row in points:[x,y,dis]
                        points = np.concatenate((points, dis), axis=1)
                    im_save_path = os.path.join(sub_save_dir, name)
                    im.save(im_save_path)
                    gd_save_path = im_save_path.replace('jpg', 'npy')
                    print(points.shape)
                    np.save(gd_save_path, points)
                        
        else:
            sub_save_dir = os.path.join(save_dir, 'test')
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
            im_list = glob(os.path.join(sub_dir, '*jpg'))
            for im_path in im_list:
                name = os.path.basename(im_path)
                print(name)
                # each row in points:[x,y]
                im, points = generate_data(im_path)
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)
