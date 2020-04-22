import sys
import json
import math
import os
import numpy as np

import movement
import common
import matplotlib.pyplot as plt



with open(sys.argv[1]) as f:
	clusters = json.load(f)

names = []
for cluster in clusters:
	for name in cluster:
		if name not in names:
			names.append(name)

movements = {}

for name1 in names:
	movements[name1] = {}
	for name2 in names:

		pairs = []
		for cluster in clusters:
			if name1 not in cluster:
				continue
			if name2 not in cluster:
				continue
			star1 = (cluster[name1]["y"], cluster[name1]["x"])
			star2 = (cluster[name2]["y"], cluster[name2]["x"])
			pairs.append((star1, star2))

		ts = []

		for i in range(len(pairs)-1):
			pair1 = pairs[i]
			for j in range(i+1, len(pairs)):
				pair2 = pairs[j]
				pi1 = pair1[0]
				pi2 = pair2[0]
				p1 = pair1[1]
				p2 = pair2[1]
				t = movement.Movement.build(pi1, pi2, p1, p2)
				ts.append(t)		

		if len(ts) >= 2:
			t,d = movement.Movement.average(ts)
			movements[name1][name2] = t.serialize()
		else:
			movements[name1][name2] = None

with open(sys.argv[2], "w") as f:
	json.dump(movements, f, indent=4)

