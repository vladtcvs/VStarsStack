from calendar import c
import numpy as np
import sys

import cfg
import common
import data
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
	summary_weight = {}
	sum_opts = {}

	for name, filename in imgs:
		print(name, filename)
		img = data.DataFrame.load(filename)

		for channel in img.get_channels():
			image, opts = img.get_channel(channel)
			if opts["encoded"]:
				continue
			if opts["weight"]:
				continue

			if channel in img.links["weight"]:
				weight_channel = img.links["weight"][channel]
				weight,_ = img.get_channel(weight_channel)
			else:
				weight = np.ones(image.shape, dtype=np.float64)

			if channel not in summary:
				summary[channel] = image.astype(np.float64)
				summary_weight[channel] = weight
			else:
				summary[channel] += image
				summary_weight[channel] += weight
			sum_opts[channel] = opts

	result = data.DataFrame()
	for channel in summary:
		print(channel)
		result.add_channel(summary[channel], channel, **(sum_opts[channel]))
		result.add_channel(summary_weight[channel], "weight-"+channel, weight=True)
		result.add_channel_link(channel, "weight-"+channel, "weight")
	result.store(out)

if __name__ == "__main__":
	run(sys.argv[1:])
