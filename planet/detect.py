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

def detect(image):
	if len(image.shape) == 3:
		gray = np.sum(image, axis=2).astype(np.float)
	else:
		gray = image.astype(np.float)

	print(gray.shape)

	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	mb = np.amax(blurred)
	blurred = blurred / mb * 255

	thr = 30
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
		if numPixels > 20:
			mask = cv2.add(mask, labelMask)

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = contours.sort_contours(cnts)[0]

	planetes = []

	# loop over the contours
	for (i, c) in enumerate(cnts):
		# draw the bright spot on the image
		(x, y, w, h) = cv2.boundingRect(c)
		((cX, cY), radius) = cv2.minEnclosingCircle(c)
		planetes.append({"x":cX, "y":cY, "size":radius})

	if len(planetes) != 1:
		print("Error: len(planetes) = %i" % (len(planetes)))
		return None
	return planetes[0]


def run(argv):
	path = argv[0]
	if len(argv) > 1:
		jsonpath = argv[1]
	else:
		jsonpath = path

	files = common.listfiles(path, ".npz")
	for name, filename  in files:
		print(name)
		image = np.load(filename)["arr_0"]

		planet = detect(image)
		if planet is None:
			print("No planet detected")
			continue
		desc = {
			"planet" : planet,
			"height" : image.shape[0],
			"width" : image.shape[1],
		}

		with open(os.path.join(jsonpath, name + ".json"), "w") as f:
			json.dump(desc, f, indent=4)

if __name__ == "__main__":
	run(sys.argv[1:])

