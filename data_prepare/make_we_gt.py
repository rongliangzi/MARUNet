import h5py
import csv
import os
import glob
import numpy as np
from tqdm import tqdm
import cv2


if __name__=='__main__':
    img_paths = []
    base_dir = '/home/datamining/Datasets/CrowdCounting/we/'
    train_dir = os.path.join(base_dir, 'train/img/')
    
    img_paths = []
    for img_path in glob.glob(os.path.join(train_dir, '*.jpg')):
        img_paths.append(img_path)
    save = True
    '''
    for img_path in tqdm(img_paths):
        csv_path = img_path.replace('img', 'den').replace('jpg', 'csv')
        img = cv2.imread(img_path)
        with open(csv_path, encoding = 'utf-8') as f:
            k = np.loadtxt(f, delimiter = ",")
                
            with h5py.File(img_path.replace('.jpg', '.h5'), 'w') as hf:
                hf['density'] = k
    '''
    test_dir = os.path.join(base_dir, 'test/')
    for cur_dir in ['104207', '200608', '200702', '202201', '500717']:
        for img_path in tqdm(glob.glob(os.path.join(test_dir + cur_dir + '/img/', '*.jpg'))):
            csv_path = img_path.replace('img', 'den').replace('jpg', 'csv')
            img = cv2.imread(img_path)
            with open(csv_path, encoding = 'utf-8') as f:
                k = np.loadtxt(f, delimiter = ",")
            with h5py.File(img_path.replace('.jpg', '.h5'), 'w') as hf:
                hf['density'] = k
            if save:
                heatmap = k/np.max(k)
                # must convert to type unit8
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.9 + img
    
                cv2.imwrite('./we_ann.jpg', superimposed_img)
                save = False