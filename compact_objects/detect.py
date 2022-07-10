from imutils import contours
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils

import sys
import json
import os

import cfg
import common
import usage

def detect(layer, debug=False):
	sources = []

	blurred = cv2.GaussianBlur(layer, (5, 5), 0)
	mb = np.amax(blurred)
	blurred = blurred / mb * 255

	thr = cfg.config["compact_objects"]["threshold"]
	thresh = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY)[1]

	if debug:
		plt.imshow(thresh, cmap="gray")
		plt.show()

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
		if numPixels >= cfg.config["compact_objects"]["minPixels"] and numPixels <= cfg.config["compact_objects"]["maxPixels"]:
			mask = cv2.add(mask, labelMask)

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	if len(cnts) == 0:
		return None
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


def process_file(filename, descfilename):
	image = common.data_load(filename)

	for channel in cfg.config["compact_objects"]["detect_channels"]:
		layer = image["channels"][channel]
		layer = layer / np.amax(layer)

		planet = detect(layer, debug=False)	
		
		if planet is not None:
			break
	else:
		print("No planet detected")
		return

	desc = {
			"compact_object"  : planet,
			"height" : image["meta"]["params"]["originalH"],
			"width"  : image["meta"]["params"]["originalW"],
	}

	with open(descfilename, "w") as f:
		json.dump(desc, f, indent=4)

def process_path(npys, descs):
	files = common.listfiles(npys, ".zip")
	for name, filename  in files:
		print(name)
		out = os.path.join(descs, name + ".json")
		process_file(filename, out)

def process(argv):
	if len(argv) > 0:
		input = argv[0]
		output = argv[1]
		if os.path.isdir(input):
			process_path(input, output)
		else:
			process_file(input, output)
	else:
		process_path([cfg.config["paths"]["npy"], cfg.config["compact_objects"]["paths"]["descs"]])


commands = {
	"*" : (process, "detect compact objects", "npy/ descs/"),
}

def run(argv):
	usage.run(argv, "compact_objects detect", commands)
