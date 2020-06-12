import matplotlib as mpl
# we cannot use remote server's GUI, so set this  
mpl.use('Agg')
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from matplotlib import cm as CM
import cv2
from tqdm import tqdm
from utils import gaussian_filter_density


# set the root to the dataset you download
max_size = 1024
root = "/home/datamining/Datasets/CrowdCounting/UCF-QNRF_ECCV18/"

# now generate the ground truth density maps
train_dir = os.path.join(root, 'Train/')
test_dir = os.path.join(root, 'Test/')
train_save = '/home/datamining/Datasets/CrowdCounting/qnrf_.1024_a/train/'
test_save = '/home/datamining/Datasets/CrowdCounting/qnrf_.1024_a/test/'
path_sets = [train_dir, test_dir]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
img_path = ''
img_save_path = ''
for img_path in tqdm(img_paths):
    mat = io.loadmat(img_path.replace('.jpg', '_ann.mat'))
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[0:2]
    # ensure the shorter side > min_size
    if img_h>img_w:
        if img_h>max_size:
            # up sample ratio is int
            ratio = max_size/img_h
            new_w, new_h = int(img_w*ratio), max_size
        else:
            ratio = 1
            new_w, new_h = img_w, img_h
    else:
        if img_w>max_size:
            ratio = max_size/img_w
            new_w, new_h = max_size, int(img_h*ratio)
        else:
            ratio = 1
            new_w, new_h = img_w, img_h
    
    img_save_path = os.path.join(train_save, img_path.split('/')[-1]) if 'rain' in img_path else os.path.join(test_save, img_path.split('/')[-1])
    
    if ratio == 1:
        cv2.imwrite(img_save_path, img)
    else:
        img_resize = cv2.resize(img, (new_w, new_h))
        cv2.imwrite(img_save_path, img_resize)
        print('img:',img_path.split('/')[-1],', img_h, img_w:',img_h, ', ', img_w, ', new_h, new_w:', new_h, ', ', new_w)
    
    k = np.zeros((new_h, new_w))
    gt = mat["annPoints"]
    #print('count annotated: ', len(gt))
    for position in gt:
        x = int(position[0]*ratio)
        y = int(position[1]*ratio)
        if x >= new_w or y >= new_h:
            continue
        k[y, x]=1
    k = gaussian_filter_density(k)
    with h5py.File(img_save_path.replace('.jpg', '.h5'), 'w') as hf:
        hf['density'] = k

'''
# see a sample

img_path = root+'Train/img_0001.jpg'
img = cv2.imread(img_path)
img_resize = cv2.resize(img, (w,h))
height, width = img.shape[0:2]
mat = io.loadmat(img_path.replace('.jpg', '_ann.mat'))
gt = mat["annPoints"]
for position in gt:
    x = int(position[0]*w/width)
    y = int(position[1]*h/height)
    cv2.circle(img_resize, (x, y), 3, (0, 0, 255), -1)
cv2.imwrite('../figs/head_center.jpg', img_resize)

img = cv2.imread(img_save_path)
# fixed gaussian filter
fixed = h5py.File(img_rsz_path.replace('.jpg','.h5'),'r')
fixed = np.asarray(fixed['density'])
heatmap = fixed/np.max(fixed)
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap*0.9+img
imgs = np.hstack([img, superimposed_img])

cv2.imwrite('../figs/qnrf_superimposed_img.jpg', imgs)

'''