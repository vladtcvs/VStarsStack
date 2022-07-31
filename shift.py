import cfg

import json
import sys
import numpy as np
import os

import projection
import common

import shift_image
import multiprocessing as mp

ncpu = max(int(mp.cpu_count())-1, 1)

from movement_sphere import Movement as ms
from movement_flat import Movement as mf

def make_shift(filename, name, name0, shifts, path):
	if name not in shifts:
		return
	if name0 not in shifts[name]:
		return
	print(name)
	if not os.path.exists(filename):
		print("skip")
		return

	t = shifts[name][name0]
	image = common.data_load(filename)
	weight = image["meta"]["params"]["weight"]

	if "weight" in img["channels"]:
		weight_layer = img["channels"]["weight"]
	else:
		weight_layer = None

	for channel in image["meta"]["channels"]:
		if channel in image["meta"]["encoded_channels"]:
			continue
		image = image["channels"][channel]
		shifted, shifted_weight = shift_image.shift_image(image, t, weight_layer, name, weight)
		common.data_add_channel(img, shifted, channel)
		common.data_add_channel(img, shifted_weight, "weight")

	common.data_store(img, os.path.join(path, name + ".zip"))

def run(argv):
	if len(argv) > 0:
		npy_dir = argv[0]
		shifts_fname = argv[1]
		shifted_dir = argv[2]
	else:
		npy_dir = cfg.config["paths"]["npy-fixed"]
		shifts_fname = cfg.config["paths"]["shifts"]
		shifted_dir = cfg.config["paths"]["shifted"]

	images = common.listfiles(npy_dir, ".zip")
	with open(shifts_fname) as f:
		data = json.load(f)
	shiftsf = data["movements"]
	if data["shift_type"] == "flat":
		Movement = mf
	elif data["shift_type"] == "sphere":
		Movement = ms
	else:
		raise Exception("Unknown shift type %s!" % data["shift_type"])
	

	shifts = {}
	names = []

	for name1 in shiftsf:
		shifts[name1] = {}
		names.append(name1)
		for name2 in shiftsf:
			if shiftsf[name1][name2] is None:
				continue
			shifts[name1][name2] = Movement.deserialize(shiftsf[name1][name2])

	# find image with minimal shift to other
	name0 = None
	mind2 = None
	maxc = None
	for name in names:
		c = len(shifts[name])
		print(name, c)
		if maxc is not None and c < maxc:
			continue
		if maxc is None or c > maxc:
			maxc = c
			mind2 = None
		d2 = 0
		for name2 in shifts[name]:
			shift = shifts[name][name2]
			d2 += shift.magnitude()

		if mind2 is None or d2 < mind2:
			mind2 = d2
			name0 = name

	print("Select:", name0, maxc)

	pool = mp.Pool(ncpu)
	pool.starmap(make_shift, [(filename, name, name0, shifts, shifted_dir) for name, filename in images])
	pool.close()

if __name__ == "__main__":
	run(sys.argv[1:])

