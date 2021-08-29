import cfg
import sys
import json
import math
import os
import numpy as np
import usage

import common
import matplotlib.pyplot as plt

percent = 60

if hasattr(cfg, "use_sphere"):
	if cfg.use_sphere:
		from movement_sphere import Movement
	else:
		from movement_flat import Movement
else:
	pass

def find_shift(pair1, pair2):
	pi1 = pair1[0]
	pi2 = pair2[0]
	p1 = pair1[1]
	p2 = pair2[1]
	t = Movement.build(pi1, pi2, p1, p2, cfg.camerad)
	return t

def run(argv):
	with open(argv[0]) as f:
		clusters = json.load(f)

	print(clusters)
	names = []
	for cluster in clusters:
		for name in cluster:
			if name not in names:
				names.append(name)

	movements = {}

	for name1 in names:
		movements[name1] = {}
		for name2 in names:
			print("%s / %s" % (name1, name2))
			pairs = []
			for cluster in clusters:
				if name1 not in cluster:
					continue
				if name2 not in cluster:
					continue
				if cfg.use_sphere:
					star1 = (cluster[name1]["lat"], cluster[name1]["lon"])
					star2 = (cluster[name2]["lat"], cluster[name2]["lon"])
				else:
					star1 = (cluster[name1]["y"], cluster[name1]["x"])
					star2 = (cluster[name2]["y"], cluster[name2]["x"])
					
				pairs.append((star1, star2))

			ts = []

			for i in range(len(pairs)-1):
				pair1 = pairs[i]
				for j in range(i+1, len(pairs)):
					pair2 = pairs[j]
					try:
						t = find_shift(pair1, pair2)
						ts.append(t)
					except:
						print("Can not find movement")
						continue

			if len(ts) >= 1:
				t = Movement.average(ts, percent)
				movements[name1][name2] = t.serialize()
			else:
				movements[name1][name2] = None
	data = {
		"movements" : movements
	}
	if cfg.use_sphere:
		data["shift_type"] = "sphere"
	else:
		data["shift_type"] = "flat"
	with open(argv[1], "w") as f:
		json.dump(data, f, indent=4)

