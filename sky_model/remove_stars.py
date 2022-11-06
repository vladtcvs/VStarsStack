import targets.stars.detect
import scipy.signal
import numpy as np

def remove_stars(image):
    size = 31
    _,mask = targets.stars.detect.detect(image)
    idx = (mask == 0)
    sidx = (mask != 0)
    nimg = np.zeros(image.shape)
    nimg[idx] = image[idx]
    nimg[sidx] = np.average(image[idx])
    filtered = scipy.signal.medfilt2d(nimg, size)
    nimg[sidx] = filtered[sidx]
    return nimg
