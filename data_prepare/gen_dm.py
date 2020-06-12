import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import cv2
import scipy.spatial
import copy

def main():
    root = "/home/datamining/Datasets/CrowdCounting/"
    sh = root+"shanghaitech/part_B_final/"
    train_dir = sh+'train_data/images/'
    img_name = 'IMG_10'
    img_path = train_dir+img_name+'.jpg'
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img = cv2.imread(img_path)
    gt = mat["image_info"][0, 0][0, 0][0]
    pts = np.array(gt)
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)
    gt_pos_img=copy.deepcopy(img)
    plt.figure(figsize=(16,12))
    for point in gt:
        cv2.circle(gt_pos_img, (int(point[0]), int(point[1])), radius=10, color=(0,0,255), thickness=2)
    plt.subplot(2,2,1)
    plt.imshow(gt_pos_img[:, :, ::-1])
    plt.title('fixed radius=10')
    
    ada_pos_img=copy.deepcopy(img)
    sigma0=[]
    reliable=[]
    for i, point in enumerate(pts):
        sigma0.append((distances[i][1]+distances[i][2]+distances[i][3])*0.1)
    colors=[]
    for i, s in enumerate(sigma0):
        a,b,c=locations[i][1:4]
        s1=(sigma0[a]+sigma0[b]+sigma0[c])/3
        if (s/s1)>1.4:
            reliable.append(0)
            color=(0,255,0)
        else:
            reliable.append(1)
            color=(0,0,255)
        colors.append(color)
        cv2.circle(ada_pos_img, (int(pts[i][0]), int(pts[i][1])), radius=int(s), color=color, thickness=2)
    plt.subplot(2,2,2)
    plt.imshow(ada_pos_img[:, :, ::-1])
    plt.title('adaptive')
    
    adjust_pos_img=copy.deepcopy(img)
    reliable=np.array(reliable)
    while (reliable-1).sum()!=0:
        for i, s in enumerate(sigma0):
            if not reliable[i]:
                for j in range(3):
                    if reliable[locations[i][j+1]]:
                        if tree.data[locations[i][j+1]][1]>pts[i][1]:
                            sigma0[i]=distances[i][j+1]*0.2
                        else:
                            sigma0[i]=distances[i][j+1]*0.4
                        reliable[i]=1
    for i, s in enumerate(sigma0):
        cv2.circle(adjust_pos_img, (int(pts[i][0]), int(pts[i][1])), radius=int(s), color=colors[i], thickness=2)
    
    plt.subplot(2,2,3)
    plt.imshow(adjust_pos_img[:, :, ::-1])
    plt.title('adjust adaptive')
    
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
    plt.savefig('figs/'+img_name+'_annotation.png')
    

if __name__=='__main__':
    main()