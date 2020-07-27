import common

import json
import sys
import random

def find_cluster(net, name, starid):
	cluster = [(name, starid)]

	changed = True
	while changed:
		changed = False
		
		for name1 in net:
			for i in net[name1]:
				for name2, j in cluster:
					if name2 in net[name1][i] and net[name1][i][name2] == j:
						if (name1, i) not in cluster:
							cluster.append((name1, i))
							changed = True
	return cluster

def normalize_cluster(cluster):
	c = {}
	for name, sid in cluster:
		if name in c:
			return False, {}
		c[name] = sid
	return True, c

def run(argv):
	with open(argv[0]) as f:
		rnet = json.load(f)

	net = {}
	for name in rnet:
		net[name] = {}
		for sid in rnet[name]:
			i = int(sid)
			net[name][i] = rnet[name][sid]

	allstars = []
	for name in net:
		for i in net[name]:
			allstars.append((name, i))

	descdir = argv[1]
	descfiles = common.listfiles(descdir, ".json")
	descs = {}
	for name, filename in descfiles:
		with open(filename) as f:
			descs[name] = json.load(f)

	clusters = []
	while len(allstars) > 0:
		name, starid = allstars[0]
		cluster = find_cluster(net, name, starid)
		res, cl = normalize_cluster(cluster)
		if res:
			clusters.append(cl)
		else:
			print("Invalid cluster", cluster)
		for name, sid in cluster:
			allstars.remove((name, sid))

	dclusters = []
	for cluster in clusters:
		dcluster = {}
		for name in cluster:
			star = cluster[name]
			y = descs[name]["main"][star]["y"]
			x = descs[name]["main"][star]["x"]
			lat = descs[name]["main"][star]["lat"]
			lon = descs[name]["main"][star]["lon"]
			size = descs[name]["main"][star]["size"]
			dcluster[name] = {
						"y" : y,
						"x" : x,
						"lat" : lat,
						"lon" : lon,
						"size" : size
					}
		if len(dcluster) >= 2:
			dclusters.append(dcluster)

	with open(argv[2], "w") as f:
		json.dump(dclusters, f, indent=4)

if __name__ == "__main__":
	run(sys.argv[1:])

