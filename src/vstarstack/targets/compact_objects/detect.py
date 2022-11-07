#
# Copyright (c) 2022 Vladislav Tsendrovskii
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

from imutils import contours
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils

import vstarstack.data
import sys
import json
import os

import vstarstack.cfg
import vstarstack.common
import vstarstack.usage

def detect(layer, debug=False):
	sources = []

	blurred = cv2.GaussianBlur(layer, (5, 5), 0)
	mb = np.amax(blurred)
	blurred = blurred / mb * 255

	thr = vstarstack.cfg.config["compact_objects"]["threshold"]
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
		if numPixels >= vstarstack.cfg.config["compact_objects"]["minPixels"] and numPixels <= vstarstack.cfg.config["compact_objects"]["maxPixels"]:
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
	image = vstarstack.data.DataFrame.load(filename)

	for channel in image.get_channels():
		layer,opts = image.get_channel(channel)
		if not opts["brightness"]:
			continue
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
	files = vstarstack.common.listfiles(npys, ".zip")
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
		process_path(vstarstack.cfg.config["paths"]["npy-fixed"], vstarstack.cfg.config["compact_objects"]["paths"]["descs"])


commands = {
	"*" : (process, "detect compact objects", "npy/ descs/"),
}

def run(argv):
	vstarstack.usage.run(argv, "compact_objects detect", commands)
