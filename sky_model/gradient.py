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
import sky_model.remove_stars

def model(image):
	shape = image.shape
	w = shape[1]
	h = shape[0]

	ws = int(w/10)
	hs = int(h/10)

	image_nostars = sky_model.remove_stars.remove_stars(image)

	left_top    = np.average(image_nostars[0:hs, 0:ws])
	left_bottom = np.average(image_nostars[h-1-hs:h-1, 0:ws])

	right_top    = np.average(image_nostars[0:hs, w-1-ws:w-1])
	right_bottom = np.average(image_nostars[h-1-hs:h-1, w-1-ws:w-1])

	bottom_k = np.array(range(h))/(h-1)
	top_k = 1-bottom_k

	right_k = np.array(range(w))/(w-1)
	left_k = 1-right_k
	
	left_top_k = top_k[:, np.newaxis] * left_k
	right_top_k = top_k[:, np.newaxis] * right_k
	left_bottom_k = bottom_k[:, np.newaxis] * left_k
	right_bottom_k = bottom_k[:, np.newaxis] * right_k

	sky = left_top * left_top_k + left_bottom * left_bottom_k + right_top * right_top_k + right_bottom * right_bottom_k
	return sky
