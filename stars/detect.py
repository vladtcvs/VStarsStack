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

import stars.detector.detector

detect = stars.detector.detector.detect_stars

def process(argv):
	path = argv[0]

	if os.path.isdir(path):
		if len(argv) > 1:
			jsonpath = argv[1]
		else:
			jsonpath = path
		files = common.listfiles(path, ".zip")

		for name, filename  in files:
			print(name)

			image = common.data_load(filename)
			stars = detect(image, debug=False)[0]
			desc = {
				"stars"  : stars,
				"height" : image["meta"]["params"]["originalH"],
				"width"  : image["meta"]["params"]["originalW"],
			}

			with open(os.path.join(jsonpath, name + ".json"), "w") as f:
				json.dump(desc, f, indent=4)
	else:
		jsonpath = argv[1]

		image = common.data_load(path)
		stars = detect(image, debug=True)[0]		
		desc = {
				"stars"  : stars,
				"height" : image["meta"]["params"]["originalH"],
				"width"  : image["meta"]["params"]["originalW"],
		}

		with open(jsonpath, "w") as f:
			json.dump(desc, f, indent=4)

commands = {
	"*" : (process, "detect stars", "npy/ stars/"),
}

def run(argv):
	usage.run(argv, "stars detect", commands)
