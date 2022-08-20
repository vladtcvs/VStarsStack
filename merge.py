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

	shape = None
	params = None

	summary = {}

	for name, filename in imgs:
		print(name)
		img = common.data_load(filename)

		if params is None:
			params = img["meta"]["params"]

		for channel in img["meta"]["channels"]:
			if channel in img["meta"]["encoded_channels"]:
				continue
			if channel not in summary:
				summary[channel] = img["channels"][channel].astype(np.float64)
			else:
				summary[channel] += img["channels"][channel]

	summary_data = common.data_create({}, params)
	for channel in summary:
		print(channel)
		common.data_add_channel(summary_data, summary[channel], channel)

	common.data_store(summary_data, out, compress=True)

if __name__ == "__main__":
	run(sys.argv[1:])
