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
import data
import cfg

import stars.detector.detector

detect = stars.detector.detector.detect_stars

def process_file(fname, jsonfile):
	image = data.DataFrame.load(fname)
	
	sources = []
	for channel in image.get_channels():
		layer, options = image.get_channel(channel)
		if not options["brightness"]:
			continue
		layer = layer / np.amax(layer)
		sources.append(layer)
	gray = sum(sources)

	stars = detect(gray, debug=False)[0]
	desc = {
		"stars"  : stars,
		"h" : image.params["h"],
		"w" : image.params["w"],
		"projection" : image.params["projection"],
		"H" : image.params["perspective_kh"] * image.params["h"],
		"W" : image.params["perspective_kw"] * image.params["w"],
		"F" : image.params["perspective_F"],
	}

	with open(jsonfile, "w") as f:
		json.dump(desc, f, indent=4)

def process_dir(path, jsonpath):
	files = common.listfiles(path, ".zip")

	for name, filename in files:
		print(name)
		process_file(filename, os.path.join(jsonpath, name + ".json"))

def process(argv):
	if len(argv) >= 2:
		path = argv[0]
		jsonpath = argv[1]
	else:
		path = cfg.config["paths"]["npy-fixed"]
		jsonpath = cfg.config["stars"]["paths"]["stars"]

	if os.path.isdir(path):
		process_dir(path, jsonpath)		
	else:
		process_file(path, jsonpath)

commands = {
	"*" : (process, "detect stars", "[npy/ stars/]"),
}

def run(argv):
	usage.run(argv, "stars detect", commands)
