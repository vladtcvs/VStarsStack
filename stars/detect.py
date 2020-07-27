from imutils import contours
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils

import sys
import json
import os

import common

percent = 0.04
min_star_pixels = 5
max_star_pixels = 144


def detect(image):
	if len(image.shape) == 3:
		gray = np.sum(image, axis=2).astype(np.float)
	else:
		gray = image.astype(np.float)

	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	mb = np.amax(blurred)
	blurred = blurred / mb * 255

	hist = np.histogram(blurred, bins=1024)
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
		
	print("Threshold = %f" % thr)

	thresh = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY)[1]

#	plt.imshow(thresh, cmap="gray")
#	plt.show()

	labels = measure.label(thresh, connectivity=2, background=0)
	mask = np.zeros(thresh.shape, dtype="uint8")

	# loop over the unique components
	for label in np.unique(labels):
		# if this is the background label, ignore it
		if label == 0:
			continue
		# otherwise, construct the label mask and count the
		# number of pixels 
		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		# if the number of pixels in the component is sufficiently
		# large, then add it to our mask of "large blobs"
		if numPixels > min_star_pixels and numPixels < max_star_pixels:
			mask = cv2.add(mask, labelMask)

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
#	print(cnts)
	cnts = contours.sort_contours(cnts)[0]

	stars = []

	# loop over the contours
	for (i, c) in enumerate(cnts):
		# draw the bright spot on the image
		(x, y, w, h) = cv2.boundingRect(c)
		((cX, cY), radius) = cv2.minEnclosingCircle(c)
		stars.append({"x":cX, "y":cY, "size":radius})

	stars.sort(key=lambda s : -s["size"])
	return stars

def run(argv):
	path = argv[0]
	if len(argv) > 1:
		jsonpath = argv[1]
	else:
		jsonpath = path

	files = common.listfiles(path, ".npy")

	for name, filename  in files:
		print(name)
		image = np.load(filename)

		stars = detect(image)
		desc = {
			"stars" : stars,
			"height" : image.shape[0],
			"width" : image.shape[1],
		}

		with open(os.path.join(jsonpath, name + ".json"), "w") as f:
			json.dump(desc, f, indent=4)

if __name__ == "__main__":
	run(argv[1:])

