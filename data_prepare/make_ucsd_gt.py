import h5py
import scipy.io as io
import os
import glob
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2
from scipy.ndimage.filters import gaussian_filter


if __name__=='__main__':
    img_paths = []
    base_dir = '/home/datamining/Datasets/CrowdCounting/vidf-cvpr/'
    
    mat_paths = []
    for mat_path in glob.glob(os.path.join(base_dir, '*.mat')):
        if 'frame' in mat_path:
            mat_paths.append(mat_path)
    
    for mat_path in mat_paths:
        img_dir = '/home/datamining/Datasets/CrowdCounting/ucsdpeds/vidf/'
        _,filename = os.path.split(mat_path)
        cur_dir = img_dir + filename.split('_frame')[0] + '.y/'
        print(cur_dir)
        mat = io.loadmat(mat_path)
        for img_path in tqdm(glob.glob(os.path.join(cur_dir, '*.png'))):
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[0:2]
            seq_id = img_path.split('.png')[0].split('_f')[-1]
            
            seq_id = int(seq_id)
            gt = mat["frame"][0, seq_id-1]["loc"][0,0]
            
            k = np.zeros((img_h, img_w))
            
            for position in gt:
                x = int(position[0])
                y = int(position[1])
                x = max(3, x)
                x = min(img_w-4, x)
                y = max(3, y)
                y = min(img_h-4, y)
                k[y, x]=1
            k = gaussian_filter(k, 3)
            with h5py.File(img_path.replace('.png', '.h5'), 'w') as hf:
                hf['density'] = k
            if seq_id == 1:
                heatmap = k/np.max(k)
                # must convert to type unit8
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.9 + img
                
                for point in gt:
                    cv2.circle(img, (int(point[0]), int(point[1])), radius=10, color=(0,0,255), thickness=2)
                    
                imgs = np.hstack([img, superimposed_img])
    
                cv2.imwrite('./ucsd_ann.jpg', imgs)