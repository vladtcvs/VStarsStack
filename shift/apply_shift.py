import cfg

import json
import sys
import numpy as np
import os

import projection.perspective
import common

import shift.shift_image
import multiprocessing as mp

ncpu = max(int(mp.cpu_count()/2)-1, 1)

from movement.sphere import Movement as ms
from movement.flat import Movement as mf

def make_shift(name, infname, image_shift, outfname):
	print(name)
	if not os.path.exists(infname) or image_shift is None:
		print("skip")
		return

	data = common.data_load(infname)
	weight = data["meta"]["params"]["weight"]

	proj = data["meta"]["params"]["projection"]
	if proj == "perspective":
		h = data["meta"]["params"]["h"]
		w = data["meta"]["params"]["w"]
		W = data["meta"]["params"]["perspective_kw"] * w
		H = data["meta"]["params"]["perspective_kh"] * h
		F = data["meta"]["params"]["perspective_F"]
		proj = projection.perspective.PerspectiveProjection(W, H, F, w, h)
	else:	
		raise Exception("Unknown projection %s" % proj)
	
	if "weight" in data["channels"]:
		weight_layer = data["channels"]["weight"]
	else:
		weight_layer = None

	for channel in data["meta"]["channels"]:
		if channel in data["meta"]["encoded_channels"]:
			continue
		image = data["channels"][channel]
		shifted, shifted_weight = shift.shift_image.shift_image(image, image_shift, proj, weight_layer, weight)
		common.data_add_channel(data, shifted, channel)
		common.data_add_channel(data, shifted_weight, "weight")

	common.data_store(data, outfname)

def run(argv):
	if len(argv) > 0:
		npy_dir = argv[0]
		shifts_fname = argv[1]
		shifted_dir = argv[2]
	else:
		npy_dir = cfg.config["paths"]["npy-fixed"]
		shifts_fname = cfg.config["paths"]["absolute-shifts"]
		shifted_dir = cfg.config["paths"]["shifted"]

	with open(shifts_fname) as f:
		data = json.load(f)
		shifts = data["movements"]

	if data["shift_type"] == "flat":
		Movement = mf
	elif data["shift_type"] == "sphere":
		Movement = ms
	else:
		raise Exception("Unknown shift type %s!" % data["shift_type"])

	for name in shifts:
		if shifts[name] is not None:
			shifts[name] = Movement.deserialize(shifts[name])

	images = common.listfiles(npy_dir, ".zip")

	for name,_ in images:
		if name not in shifts:
			shifts[name] = None

	pool = mp.Pool(ncpu)
	pool.starmap(make_shift, [(name, filename, shifts[name], os.path.join(shifted_dir, name + ".zip")) for name, filename in images])
	pool.close()
