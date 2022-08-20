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
import stars.detect

def model(image, mask):
	shape = image.shape
	w = shape[1]
	h = shape[0]

	ws = int(w/8)
	hs = int(h/8)

	left_top    = np.average(image[0:hs, 0:ws])
	left_bottom = np.average(image[h-1-hs:h-1, 0:ws])

	right_top    = np.average(image[0:hs, w-1-ws:w-1])
	right_bottom = np.average(image[h-1-hs:h-1, w-1-ws:w-1])

	sky = np.zeros(shape)
	for y in range(h):
		ky = y / (h-1)
		for x in range(w):
			kx = x / (w-1)
			pix = (left_top * (1-ky) + left_bottom * ky) * (1-kx) + (right_top * (1-ky) + right_bottom * ky) * kx
			sky[y,x] = pix
	return sky
