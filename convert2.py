import numpy as np
import sys
import cv2
import json
import normalize
import imageio

import matplotlib.pyplot as plt

import common

power = 1

path    = sys.argv[1]
channel = sys.argv[2]

data = common.data_load(path)

if channel == "RGB":
	r = data["channels"]["R"]
	g = data["channels"]["G"]
	b = data["channels"]["B"]

	rgb = np.zeros((r.shape[0], r.shape[1], 3))
	rgb[:,:,0] = r
	rgb[:,:,1] = g
	rgb[:,:,2] = b
	data = rgb
else:
	data = data["channels"][channel]

img = np.clip(data, -1e6, 1e6)

minv = np.amin(img)
if minv < 0:
	img -= minv

amax = np.amax(img)
img /= amax
img = np.power(img, power)

amax = np.amax(img)
img = img/amax*65535
img = np.clip(img, 0, 65535).astype('uint16')

imageio.imwrite(sys.argv[3], img)
