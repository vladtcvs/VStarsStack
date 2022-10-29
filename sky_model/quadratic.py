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

	sz = 0.01

	ws = int(w*sz+0.5)
	hs = int(h*sz+0.5)

	center_w = int(w/2+0.5)
	center_h = int(h/2+0.5)
	ws_half = int(ws/2)
	hs_half = int(hs/2)

	image_nostars = sky_model.remove_stars.remove_stars(image)

	left_top      = np.average(image_nostars[0:hs, 0:ws])
	left_center   = np.average(image_nostars[center_h-hs_half:center_h+hs_half+1, 0:ws])
	left_bottom   = np.average(image_nostars[h-1-hs:h-1, 0:ws])
	center_top    = np.average(image_nostars[0:hs, center_w-ws_half:center_w+ws_half+1])
	right_top     = np.average(image_nostars[0:hs, w-1-ws:w-1])
	right_center  = np.average(image_nostars[center_h-hs_half:center_h+hs_half+1, w-1-ws:w-1])
	right_bottom  = np.average(image_nostars[h-1-hs:h-1, w-1-ws:w-1])
	center_bottom = np.average(image_nostars[h-1-hs:h-1, center_w-ws_half:center_w+ws_half+1])

	center_center = (left_center + center_top + right_center + center_bottom) / 4

	vertical_down_i   = np.array(range(h))/(h-1)
	vertical_up_i     = 1 - vertical_down_i
	vertical_arc_i    = 4*vertical_down_i * vertical_up_i

	horizontal_right_i = np.array(range(w))/(w-1)
	horizontal_left_i  = 1-horizontal_right_i
	horizontal_arc_i   = 4*horizontal_left_i * horizontal_right_i

	down_right_f = vertical_down_i[:, np.newaxis] * horizontal_right_i
	down_arc_f   = vertical_down_i[:, np.newaxis] * horizontal_arc_i
	down_left_f  = vertical_down_i[:, np.newaxis] * horizontal_left_i

	arc_right_f = vertical_arc_i[:, np.newaxis] * horizontal_right_i
	arc_arc_f   = vertical_arc_i[:, np.newaxis] * horizontal_arc_i
	arc_left_f  = vertical_arc_i[:, np.newaxis] * horizontal_left_i

	up_right_f = vertical_up_i[:, np.newaxis] * horizontal_right_i
	up_arc_f   = vertical_up_i[:, np.newaxis] * horizontal_arc_i
	up_left_f  = vertical_up_i[:, np.newaxis] * horizontal_left_i

	a0 = right_bottom
	b0 = left_bottom
	c0 = center_bottom - (a0+b0)/2

	a1 = right_top
	b1 = left_top
	c1 = center_top - (a1+b1)/2

	a2 = right_center - (a0+a1)/2
	b2 = left_center - (b0+b1)/2
	c2 = center_center - (a2+b2)/2 - (c0+c1)/2 - (a0+b0+a1+b1)/4

	sky0 = down_right_f * a0 + down_left_f * b0 + down_arc_f * c0
	sky1 = up_right_f * a1 + up_left_f * b1 + up_arc_f * c1
	sky2 = arc_right_f * a2 + arc_left_f * b2 + arc_arc_f * c2
	
	sky = sky0 + sky1 + sky2

	return sky
