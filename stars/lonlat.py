import math
import sys
import json
import os

import cfg
import usage
import projection.perspective
import common

def process_file(jsonfile):
	with open(jsonfile) as f:
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
	with open(jsonfile, "w") as f:
		json.dump(desc, f, indent=4)

def process_dir(path):
	descs = common.listfiles(path, ".json")
	for name, filename in descs:
		print(name)
		process_file(filename)

def process(argv):
	if len(argv) >= 1:
		path = argv[0]
	else:
		path = cfg.config["stars"]["paths"]["stars"]

	if os.path.isdir(path):
		process_dir(path)		
	else:
		process_file(path)

commands = {
	"*" : (process, "fill longitude and latitude for stars", "stars/"),
}

def run(argv):
	usage.run(argv, "stars lonlat", commands)
