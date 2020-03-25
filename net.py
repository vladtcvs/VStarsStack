import sys
import os
import json

import common

jsondir = sys.argv[1]
out = sys.argv[2]

imagesfiles = common.listfiles(jsondir, ".json")

names = []

images = {}
for name,filename in imagesfiles:
	names.append(name)
	with open(filename) as f:
		images[name] = json.load(f)

net = {}
for name in names:
	net[name] = {}
	image = images[name]
	stars = image["main"]
	for name2 in names:
		net[name][name2] = {}

		for i in range(len(stars)) :
			net[name][name2][i] = stars[i]["matches"][name2]

for name1 in names:
	for name2 in names:
		m1 = net[name1][name2]
		m2 = net[name2][name1]
		for i in m1:
			match = m1[i]
			if match is None:
				continue
			match2 = m2[match]
			if match2 is None:
				m2[match] = i
				continue
			if match2 != i:
				print("Error matching: %s %i <-> %s %i <-> %s %i" % (name1, i, name2, match, name1, match2))
				m1[i] = None
				m2[match] = None

with open(out, "w") as f:
	json.dump(net, f, indent=4)

