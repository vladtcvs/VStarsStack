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
import cfg

import stars.detector.detector

detect = stars.detector.detector.detect_stars

def process_file(fname, jsonfile):
	image = common.data_load(fname)
	
	sources = []
	for channel in image["channels"]:
		if channel in image["meta"]["encoded_channels"]:
			continue
		if channel == "weight":
			continue
		layer = image["channels"][channel]
		layer = layer / np.amax(layer)
		sources.append(layer)
	gray = sum(sources)

	stars = detect(gray, debug=False)[0]
	desc = {
		"stars"  : stars,
		"h" : image["meta"]["params"]["h"],
		"w" : image["meta"]["params"]["w"],
		"projection" : image["meta"]["params"]["projection"],
		"H" : image["meta"]["params"]["perspective_kh"] * image["meta"]["params"]["h"],
		"W" : image["meta"]["params"]["perspective_kw"] * image["meta"]["params"]["w"],
		"F" : image["meta"]["params"]["perspective_F"],
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
