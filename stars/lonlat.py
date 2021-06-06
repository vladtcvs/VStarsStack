import math
import sys
import json

import cfg
import usage
import projection
import common

def process(argv):
	descs = common.listfiles(argv[0], ".json")
	for name, filename in descs:
		print(name)
		with open(filename) as f:
			desc = json.load(f)
		h = desc["height"]
		w = desc["width"]
		proj = projection.Projection(cfg.camerad["W"], cfg.camerad["H"], cfg.camerad["F"], w, h)
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
			
