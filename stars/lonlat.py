import math
import sys
import json

import cfg
import usage
import projection.perspective
import common

def process(argv):
	descs = common.listfiles(argv[0], ".json")
	for name, filename in descs:
		print(name)
		with open(filename) as f:
			desc = json.load(f)
		h = desc["h"]
		w = desc["w"]
		proj = desc["projection"]
		if proj == "perspective":
			W = desc["W"]
			H = desc["H"]
			F = desc["F"]
			proj = projection.perspective.PerspectiveProjection(W, H, F, w, h)
		else:
			raise Exception("Unknown projection %s" % projection)
		
		if "stars" in desc:
			for star in desc["stars"]:
				x = star["x"]
				y = star["y"]
				lat, lon = proj.project(y, x)
				star["lon"] = lon
				star["lat"] = lat
		with open(filename, "w") as f:
			json.dump(desc, f, indent=4)

commands = {
	"*" : (process, "fill longitude and latitude for stars", "stars/"),
}

def run(argv):
	usage.run(argv, "stars lonlat", commands)
