import numpy as np
import sys

import cfg
import common
import matplotlib.pyplot as plt

def run(argv):
	if len(argv) > 0:
		path_images = argv[0]
		out = argv[1]
	else:
		path_images = cfg.config["paths"]["shifted"]
		out = cfg.config["paths"]["output"]

	imgs = common.listfiles(path_images, ".zip")

	images = {}
	shape = None
	params = None

	for name, filename in imgs:
		img = common.data_load(filename)

		if params is None:
			params = img["meta"]["params"]

		for channel in img["meta"]["channels"]:
			if channel in img["meta"]["encoded_channels"]:
				continue
			if channel not in images:
				images[channel] = []
			images[channel].append(img["channels"][channel])

			if shape is None:
				shape = img["channels"][channel].shape

	summary = common.data_create({}, params)
	for channel in images:
		common.data_add_channel(summary, sum(images["channel"], channel))

	common.data_store(summary, out)

if __name__ == "__main__":
	run(sys.argv[1:])

