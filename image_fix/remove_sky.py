import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import sys
import cv2
import usage
import os
import common
import multiprocessing as mp

ncpu = 11
k = 4

sky_blur = 351

def detect(image):
	if len(image.shape) == 3:
		gray = np.sum(image, axis=2)
	else:
		gray = image
	gray = np.float32(gray)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	mb = np.amax(blurred)
	blurred = blurred / mb * 255

	percent = 0.5

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

	mask = cv2.threshold(blurred, thr, 1.0, cv2.THRESH_BINARY)[1]
	return mask

def fun(l):
	return 2/(l**2+1) - 1

def interp(y, x, h, w):
	L = ((w/2)**2+(h/2)**2)**0.5
	x -= w/2
	y -= h/2
	l = (x**2+y**2)**0.5
	return fun(l/L)

def skymodel_gauss(image, stars_mask):
	shape = image.shape
	sky = np.zeros(shape)
	idx  = (stars_mask==0)
	nidx = (stars_mask!=0)
	sky[idx]  = image[idx]
	average   = np.mean(sky, axis=(0,1))
	sky[nidx] = average
	sky = cv2.GaussianBlur(sky, (sky_blur, sky_blur), 0)
	return sky


def remove_sky(name, infname, outfname, method):
	print(name)
	image = np.load(infname)["arr_0"]
	shape = image.shape

	mask = detect(image)

	if method == "gauss":
		sky = skymodel_gauss(image, mask)
	else:
		sky = skymodel_gauss(image, mask)

	if len(shape) == 3:
		image[:,:,0:3] = image[:,:,0:3] - sky[:,:,0:3]

	np.savez_compressed(outfname, image)

def process_file(argv):
	infname = argv[0]
	outfname = argv[1]
	method = argv[2]
	name = os.path.splitext(os.path.basename(infname))[0]
	remove_sky(name, infname, outfname, method)

def process_dir(argv):
	inpath = argv[0]
	outpath = argv[1]
	method = argv[2]
	files = common.listfiles(inpath, ".npz")
	pool = mp.Pool(ncpu)
	pool.starmap(remove_sky, [(name, fname, os.path.join(outpath, name + ".npz"), method) for name, fname in files])
	pool.close()

commands = {
	"file" : (process_file, "process single file", "input.file output.file <method>"),
	"path" : (process_dir,  "process all files in dir", "input_path/ output_path/ <method>"),
}

def run(argv):
	usage.run(argv, "image-fix remove-sky", commands, "Methods: gauss")

