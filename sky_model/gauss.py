import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import sys
import cv2
import usage
import os
import common
import multiprocessing as mp

import cfg

def model(image, mask):
	shape = image.shape
	sky_blur = int(shape[1]/4)*2+1

	sky = np.zeros(shape)
	idx  = (mask==0)
	nidx = (mask!=0)
	sky[idx]  = image[idx]
	average   = np.mean(sky, axis=(0,1))
	sky[nidx] = average
	sky = cv2.GaussianBlur(sky, (sky_blur, sky_blur), 0)
	return sky
