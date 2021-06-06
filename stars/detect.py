from imutils import contours
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import math
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter

import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import skimage.measure as measure

import usage
import sys
import json
import os

import normalize
import common

percent = 0.01
min_star_pixels = 4
max_star_pixels = 50

basesky_size = 50

r = 2
ts = 10
template = np.zeros((2*ts+1, 2*ts+1),dtype=np.float32)
for x in range(2*ts+1):
	for y in range(2*ts+1):
		px = x - ts
		py = y - ts
		pr = (px**2 + py**2)**0.5
		template[y][x] = math.exp(-pr**2 / (2*r**2))

def find_stars_from_threshold(starsimage, debug=False):
	
	shape = starsimage.shape
	labels = measure.label(starsimage, connectivity=2, background=0)
	mask = np.zeros(shape, dtype="uint8")

	# loop over the unique components
	for label in np.unique(labels):
		# if this is the background label, ignore it
		if label == 0:
			continue
		# otherwise, construct the label mask and count the
		# number of pixels 
		labelMask = np.zeros(shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		# if the number of pixels in the component is sufficiently
		# large, then add it to our mask of "large blobs"
		if numPixels >= min_star_pixels and numPixels <= max_star_pixels:
			mask = cv2.add(mask, labelMask)

	if debug:
		plt.imshow(mask, cmap="gray")
		plt.show()

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = contours.sort_contours(cnts)[0]

	stars = []

	if debug:
		image = starsimage / np.amax(starsimage)
	# loop over the contours
	for (i, c) in enumerate(cnts):
		(x, y, w, h) = cv2.boundingRect(c)
		((cX, cY), radius) = cv2.minEnclosingCircle(c)
		stars.append({"x":cX, "y":cY, "size":radius})
		if debug:
			cv2.circle(image, (int(cX), int(cY)), 1+int(radius*5), (1, 0, 0), 1)
	if debug:
		plt.imshow(image/np.amax(image))
		plt.show()

	stars.sort(key=lambda s : -s["size"])
	return stars


def detect_by_template(image, debug=False):
	if len(image.shape) == 3:
		gray = np.sum(image, axis=2).astype(np.float)
	else:
		gray = image.astype(np.float)

	sz = int(min(gray.shape[0], gray.shape[1])/(basesky_size*2))*2+1

	basesky = cv2.GaussianBlur(gray, (sz, sz), 0)
	sky = cv2.GaussianBlur(gray, (3,3), 0)

	nosky = sky - basesky
	amin = np.amin(nosky)
	amax = np.amax(nosky)
	nosky = (nosky - amin) / (amax - amin)

	matches = cv2.matchTemplate(nosky.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)

	stars = matches > 0.75

#	plt.imshow(maximum_filter(stars, size=(6,6)), cmap="gray")
#	plt.imshow(stars, cmap="gray")
#	plt.show()
	return find_stars_from_threshold(stars, debug)


def detect_by_brightness(image, debug=False):
	if len(image.shape) == 3:
		gray = np.sum(image, axis=2).astype(np.float)
	else:
		gray = image.astype(np.float)

	if debug:
		plt.imshow(gray, cmap="gray")
		plt.show()

	sky = cv2.GaussianBlur(gray, (5,5), 0)
	blurred = cv2.GaussianBlur(gray, (101,101), 0)
	sky -= blurred

	sky /= np.amax(sky)

	if debug:
		plt.imshow(sky, cmap="gray")
		plt.show()

	hist = np.histogram(sky, bins=1024)
	nums = list(hist[0])
	bins = list(hist[1])
	nums.reverse()
	bins.reverse()

	total = sum(nums)
	maxp = total * percent / 100
	summ = 0
	for i in range(1024):
		thr = bins[i]
		c = nums[i]
		summ += c
		if summ >= maxp:
			break

	stars = cv2.threshold(sky, thr, 1, cv2.THRESH_BINARY)[1]

	if debug:
		plt.imshow(stars, cmap="gray")
		plt.show()

	return find_stars_from_threshold(stars, debug)

def get_areas(mask, minsize=None, maxsize=None):
	shape = mask.shape
	labels = measure.label(mask, connectivity=1, background=0)
	mask = np.zeros(shape)
	# loop over the unique components
	uqlabels = np.unique(labels)
#	print(len(uqlabels))
	for label in uqlabels:
		# if this is the background label, ignore it
		if label == 0:
			continue
		# otherwise, construct the label mask and count the
		# number of pixels 
		labelMask = np.zeros(shape)
		labelMask[labels == label] = 1
		numPixels = cv2.countNonZero(labelMask)
		# if the number of pixels in the component is sufficiently
		# large, then add it to our mask of "large blobs"
		if minsize is not None and numPixels < minsize:
			continue
		if maxsize is not None and numPixels > maxsize:
			continue
		mask = cv2.add(mask, labelMask)
	return mask


def detect_by_bright_areas_mask(image, debug=False):
	thr_size = 301
	blur_size = 21
	side=80
	shape = image.shape

	image = image / np.amax(image)

	if len(image.shape) == 3:
		gray = np.sum(image, axis=2).astype(np.float)
	else:
		gray = image.astype(np.float)

	gray = gray / np.amax(gray)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)

	bright_thr = filters.threshold_local(gray, block_size=thr_size, offset=-0.05)
	bright_mask = gray > bright_thr

	mask = get_areas(bright_mask, 150, None)
	mask = cv2.GaussianBlur(mask, (101, 101), 0)

	resmask = (mask == 0)

	resmask[:,:side] = 0
	resmask[:side,:] = 0
	resmask[:,-side:] = 0
	resmask[-side:,:] = 0

	result = bright_mask * resmask

	print(debug)
	if debug:
		fig, axs = plt.subplots(ncols=3, nrows=2)
		axs[0,0].imshow(image, cmap="gray")
		axs[0,0].set_title("original image")

		axs[0,1].imshow(bright_thr, cmap="gray")
		axs[0,1].set_title("brightness local threshold")

		axs[0,2].imshow(bright_mask, cmap="gray")
		axs[0,2].set_title("brightness mask - greater than threshold")

		axs[1,0].imshow(mask, cmap="gray")
		axs[1,0].set_title("mask")

		axs[1,1].imshow(resmask, cmap="gray")
		axs[1,1].set_title("resmask")

		axs[1,2].imshow(result, cmap="gray")
		axs[1,2].set_title("result")
		plt.show()


	return find_stars_from_threshold(result, debug)


detect = detect_by_bright_areas_mask

def process(argv):
	debug = False
	path = argv[0]
	if len(argv) > 1:
		jsonpath = argv[1]
	else:
		jsonpath = path

	if os.path.isdir(path):
		files = common.listfiles(path, ".npz")

		for name, filename  in files:
			print(name)
			image = np.load(filename)["arr_0"]

			if len(image.shape) == 3:
				image = normalize.normalize(image)
			stars = detect(image, debug)
			desc = {
				"stars" : stars,
				"height" : image.shape[0],
				"width" : image.shape[1],
			}

			with open(os.path.join(jsonpath, name + ".json"), "w") as f:
				json.dump(desc, f, indent=4)
	else:
		image = np.load(path)["arr_0"]

		stars = detect(image, debug=True)
		desc = {
			"stars" : stars,
			"height" : image.shape[0],
			"width" : image.shape[1],
		}

		with open(jsonpath, "w") as f:
			json.dump(desc, f, indent=4)

commands = {
	"*" : (process, "detect stars", "npy/ stars/"),
}

def run(argv):
	usage.run(argv, "stars detect", commands)

