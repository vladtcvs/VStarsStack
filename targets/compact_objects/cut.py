import common
import numpy as np
import sys
import os
import json
import cfg

def run(argv):
	npypath = argv[0]
	jsonpath = argv[1]
	cutpath = argv[2]
	if len(argv) > 3:
		size = int(argv[3])
	else:
		size = cfg.config["compact_objects"]["cutsize"]

	files = common.listfiles(jsonpath, ".json")
	for name, filename in files:
		with open(filename) as f:
			detection = json.load(f)
		


	for name, filename in files:
		print(name)
		with open(filename) as f:
			detection = json.load(f)
		imagename = os.path.join(npypath, name + ".zip")

		try:
			image = common.data_load(imagename)
		except:
			print("Can not load ", name)
			continue

		x = int(detection["compact_object"]["x"])
		y = int(detection["compact_object"]["y"])
		left  = int(x - size/2)
		right = left+size
		top   = int(y - size/2)
		bottom = top+size

		for channel in list(image["channels"].keys()):
			if channel in image["meta"]["encoded_channels"]:
				common.data_remove_channel(image, channel)
				continue
			img = image["channels"][channel]
			img = img[top:bottom, left:right]

			if img.shape[0] != size or img.shape[1] != size:
				common.data_remove_channel(image, channel)
				continue
			common.data_add_channel(image, img, channel)
		
		outname = os.path.join(cutpath, name + ".zip")
		common.data_store(image, outname)
