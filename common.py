import os
import math
import imageio
from skimage.color import rgb2gray
import cv2
from skimage import data, img_as_float
from skimage import exposure
import numpy as np

import zipfile
import json

def getpixel_linear(img, y, x):
	xm = math.floor(x)
	xmc = 1-(x-xm)
	xp = math.ceil(x)
	xpc = 1-(xp-x)
	ym = math.floor(y)
	ymc = 1-(y-ym)
	yp = math.ceil(y)
	ypc = 1-(yp-y)

	c = xmc + xpc
	xmc /= c
	xpc /= c

	c = ymc + ypc
	ymc /= c
	ypc /= c

	if ym < 0 or xm < 0 or xp >= img.shape[1] or yp >= img.shape[0]:
		if len(img.shape) == 2:
			return False, 0
		else:
			return False, np.zeros((img.shape[2],))

	imm = img[ym][xm]
	imp = img[ym][xp]
	ipm = img[yp][xm]
	ipp = img[yp][xp]

	return True, imm * ymc*xmc + imp * ymc*xpc + ipm * ypc*xmc + ipp * ypc*xpc

def getpixel_none(img, y, x):
	x = round(x)
	y = round(y)
	
	if y < 0 or x < 0 or x >= img.shape[1] or y >= img.shape[0]:
		if len(img.shape) == 2:
			return False, 0
		else:
			return False, np.zeros((img.shape[2],))

	return True, img[y][x]


def getpixel(img, y, x, interpolate=True):
	if interpolate:
		return getpixel_linear(img, y, x)
	return getpixel_none(img, y, x)

def listfiles(path, ext=None):
	images = []
	for f in os.listdir(path):
		filename = os.path.abspath(os.path.join(path, f))
		if not os.path.isfile(filename):
			continue
		if (ext is not None) and (f[-len(ext):].lower() != ext):
			continue

		name = os.path.splitext(f)[0]
		images.append((name, filename))
	images.sort(key=lambda item : item[0])
	return images

def length(vec):
	return (vec[0]**2+vec[1]**2)**0.5

def norm(vec):
	l = (vec[0]**2+vec[1]**2)**0.5
	return (vec[0] / l, vec[1] / l)	

def prepare_image_for_model(image):
	image = image[:,:,0:3]
	image = rgb2gray(image)
	image = cv2.GaussianBlur(image, (3, 3), 0)
	am = np.amax(image)
	if am >= 1:
		image /= am
	image = exposure.equalize_hist(image)
	return image

def data_create(tags = {}, params = {}):
	data = {
		"channels" : {},
		"meta" : {
			"channels" : [],
			"tags" : tags,
			"params" : params,
		},
	}
	return data

def data_add_channel(data, channel, name):
	data["channels"][name] = channel
	data["meta"]["channels"].append(name)
	return data

def data_store(data, output):
	with zipfile.ZipFile(output, "w") as zf:
		with zf.open("meta.json", "w") as f:
			f.write(bytes(json.dumps(data["meta"], indent=4, ensure_ascii=False), 'utf8'))
			#json.dump(data["meta"], f, indent=4, ensure_ascii=False)
		for channel in data["channels"]:
			with zf.open(channel+".npy", "w") as f:
				np.save(f, data["channels"][channel])

def data_load(input):
	with zipfile.ZipFile(input, "r") as zf:
		with zf.open("meta.json", "r") as f:
			meta = json.load(f)
			data = data_create(meta["tags"], meta["params"])

		for channel in meta["channels"]:
			with zf.open(channel+".npy", "r") as f:
				data_add_channel(data, np.load(f), channel)
	return data
