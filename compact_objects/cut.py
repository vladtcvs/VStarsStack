import common
import numpy as np
import sys
import os
import json

def run(argv):
	npypath = argv[0]
	jsonpath = argv[1]
	cutpath = argv[2]
	if len(argv) > 3:
		cutmul = float(argv[3])
	else:
		cutmul = 2

	files = common.listfiles(jsonpath, ".json")
	rs = []
	for name, filename in files:
		with open(filename) as f:
			detection = json.load(f)
		
		r = int(detection["compact_object"]["size"]+1)
		rs.append(r)

	r = max(rs)

	for name, filename in files:
		print(name)
		with open(filename) as f:
			detection = json.load(f)
		imagename = os.path.join(npypath, name + ".zip")

		image = common.data_load(imagename)

		x = int(detection["compact_object"]["x"])
		y = int(detection["compact_object"]["y"])
		left  = int(x - r*cutmul)
		right = int(x + r*cutmul)
		top   = int(y - r*cutmul)
		bottom = int(y + r*cutmul)

		for channel in image["channels"]:
			if channel in image["meta"]["encoded_channel"]:
				continue
			img = image["channels"][channel]
			img = img[top:bottom, left:right]
			common.data_add_channel(image, img, channel)
		
		outname = os.path.join(cutpath, name + ".zip")
		common.data_store(image, outname)
