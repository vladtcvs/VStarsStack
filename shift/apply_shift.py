import cfg

import json
import sys
import numpy as np
import os

import projection.perspective
import common
import data

import shift.shift_image
import multiprocessing as mp

ncpu = max(int(mp.cpu_count())-1, 1)

from movement.sphere import Movement as ms
from movement.flat import Movement as mf

def make_shift(name, infname, image_shift, outfname):
	print(name)
	if not os.path.exists(infname) or image_shift is None:
		print("skip")
		return

	dataframe = data.DataFrame.load(infname)

	proj = dataframe.params["projection"]
	if proj == "perspective":
		h = dataframe.params["h"]
		w = dataframe.params["w"]
		W = dataframe.params["perspective_kw"] * w
		H = dataframe.params["perspective_kh"] * h
		F = dataframe.params["perspective_F"]
		proj = projection.perspective.Projection(W, H, F, w, h)
	else:	
		raise Exception("Unknown projection %s" % proj)

	for channel in dataframe.get_channels():
		image, opts = dataframe.get_channel(channel)
		if opts["weight"]:
			continue
		if opts["encoded"]:
			continue

		weight_channel = None
		if channel in dataframe.links["weight"]:
			weight_channel = dataframe.links["weight"][channel]
		
		if weight_channel:
			weight,_ = dataframe.get_channel(weight_channel)
		else:
			weight = np.ones(image.shape)*1

		shifted, shifted_weight = shift.shift_image.shift_image(image, image_shift, proj, weight)
		dataframe.add_channel(shifted, channel, **opts)
		dataframe.add_channel(shifted_weight, weight_channel, weight=True)
		dataframe.add_channel_link(channel, weight_channel, "weight")

	dataframe.store(outfname)

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
