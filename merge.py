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

	imgs = common.listfiles(path_images, ".npz")

	images = {}
	shape = None

	for name, filename in imgs:
		image = np.load(filename)["arr_0"]
		if shape is None:
			shape = image.shape
			print(shape)
			break

	summary = np.zeros(shape)
	for name, filename in imgs:
		print(name)
		image = np.load(filename)["arr_0"]
		summary = summary + image

	np.savez_compressed(out, summary)

if __name__ == "__main__":
	run(sys.argv[1:])

