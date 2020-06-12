import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.spatial


def gaussian_filter_density(gt, min_size=0, max_size=0):
    #print('image shape(h, w):', gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2.  # case: 1 point
        if min_size !=0 and max_size!=0:
            sigma = max(min(max_size,sigma),min_size)
        density += ndimage.filters.gaussian_filter(pt2d, sigma)
    #print('density map count: ', density.sum())
    return density


def gaussian_filter_density_rsz(gt, resize_shape):
    print('image shape(h, w):', gt.shape)
    density = np.zeros(resize_shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(resize_shape, dtype=np.float32)
        row = int(pt[1]*resize_shape[0]/gt.shape[0])
        col = int(pt[0]*resize_shape[1]/gt.shape[1])
        pt2d[row, col] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('density map count: ', density.sum())
    return density
