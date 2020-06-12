import h5py
import numpy as np
import cv2


if __name__ == '__main__':
    for i in range(1,31):
        img = cv2.imread('/home/datamining/Datasets/CrowdCounting/bg/bg_{}.jpg'.format(i))
        h, w = img.shape[0:2]
        with h5py.File('/home/datamining/Datasets/CrowdCounting/bg/bg_{}.h5'.format(i), 'w') as f:
            f['density'] = np.zeros((h, w))