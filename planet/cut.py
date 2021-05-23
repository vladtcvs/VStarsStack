import common
import numpy as np
import sys
import os
import json

def run(argv):
	npypath = argv[0]
	jsonpath = argv[1]
	cutpath = argv[2]

	files = common.listfiles(jsonpath, ".json")
	rs = []
	for name, filename in files:
		with open(filename) as f:
			detection = json.load(f)
		
		r = int(detection["planet"]["size"]+1)
		rs.append(r)

	r = max(rs)

	for name, filename in files:
		with open(filename) as f:
			detection = json.load(f)
		imagename = os.path.join(npypath, name + ".npz")
		image = np.load(imagename)["arr_0"]
		
		x = int(detection["planet"]["x"])
		y = int(detection["planet"]["y"])
		left  = x - r*2
		right = x + r*2 
		top   = y - r*2
		bottom = y + r*2

		image = image[top:bottom, left:right]
		
		outname = os.path.join(cutpath, name + ".npz")
		np.savez_compressed(outname, image)

