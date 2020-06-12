import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy.spatial
from tqdm import tqdm
import cv2


# set the root to the dataset you download
root = "/home/datamining/Datasets/CrowdCounting/"

dir_dict = dict()
dir_dict['sha'] = root + "shanghaitech/part_A_final/"
dir_dict['shb'] = root + "shanghaitech/part_B_final/"

dataset = 'sha'

train_dir = os.path.join(dir_dict[dataset], 'train_data/images/')
test_dir = os.path.join(dir_dict[dataset], 'test_data/images/')
path_sets = [train_dir, test_dir]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in tqdm(img_paths):
    mat_path = img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_')
    mat = io.loadmat(mat_path)
    im = Image.open(img_path)
    im_w, im_h = im.size
    
    points = mat["image_info"][0, 0][0, 0][0].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    print(points.shape)
    if dataset=='sha':
        tree = scipy.spatial.KDTree(points.copy())
        distances, locations = tree.query(points, k=4)
        # calculate the nearest 3 people 
        sigma = np.mean(distances[:,1:]*0.3, axis=1).reshape(-1,1)
        points = np.concatenate((points, sigma), axis=1)
        print(distances[0][0])
        print(points.shape)
        gd_save_path = mat_path.replace('.mat', '.npy')
        np.save(gd_save_path, points)
    if dataset=='shb':
        k = gaussian_filter(k, 15, truncate=4.0)
    