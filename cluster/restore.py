import math
import sys
import json

import cfg

import projection
import common

def run(argv):
	infilename = argv[0]
	if len(argv) == 1:
		outfilename = infilename
	else:
		outfilename = argv[1]

	proj = projection.Projection(cfg.camerad["W"], cfg.camerad["H"], cfg.camerad["F"], cfg.camerad["w"], cfg.camerad["h"])
	with open(infilename) as f:
		clusters = json.load(f)
	for cluster in clusters:
		for name in cluster:
			x = cluster[name]["x"]
			y = cluster[name]["y"]
			lat, lon = proj.project(y, x)
			cluster[name]["lon"] = lon
			cluster[name]["lat"] = lat
	with open(outfilename, "w") as f:
		json.dump(clusters, f, indent=4)

