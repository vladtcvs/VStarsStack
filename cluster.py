import common

import json
import sys
import random

# `found` changes during calls!
def find_cluster(net, name, starid, found):
	if starid is None:
		return
	if (name, starid) in found:
		return []
	found.append((name, starid))
	for name2 in net[name]:
		sid = net[name][name2][str(starid)]
		if sid != None:
			find_cluster(net, name2, sid, found)
	cluster = {}
	for name, sid in found:
		if name in cluster and str(sid) != str(cluster[name]):
			print("Error!!!!", name, str(sid), str(cluster[name]))
		cluster[name] = sid
	return cluster

def clean_net(net):
	for name in list(net.keys()):
		for name2 in list(net[name].keys()):
			if len(net[name][name2]) == 0:
				net[name].pop(name2)
		if len(net[name]) == 0:
			net.pop(name)


def delete_cluster(net, cluster):
	for name in cluster:
		star = str(cluster[name])
		if name not in net:
			continue
		for name2 in net[name].keys():
			if star in net[name][name2]:
				net[name][name2].pop(star)
	clean_net(net)

def get_random_star(net):
	name1 = random.sample(net.keys(), 1)[0]
	name2 = random.sample(net[name1].keys(), 1)[0]
	sid = random.sample(net[name1][name2].keys(), 1)[0]
	return name1, sid
	

with open(sys.argv[1]) as f:
	net = json.load(f)

descdir = sys.argv[2]
descfiles = common.listfiles(descdir, ".json")
descs = {}
for name, filename in descfiles:
	with open(filename) as f:
		descs[name] = json.load(f)

clusters = []
while len(net) > 0:
#	print("Find cluster")
	name, starid = get_random_star(net)
	cluster = find_cluster(net, name, starid, [])
	clusters.append(cluster)
	delete_cluster(net, cluster)

dclusters = []
for cluster in clusters:
	dcluster = {}
	for name in cluster:
		star = cluster[name]
		y = descs[name]["main"][star]["y"]
		x = descs[name]["main"][star]["x"]
		size = descs[name]["main"][star]["size"]
		dcluster[name] = {"y" : y, "x" : x, "size" : size}
	dclusters.append(dcluster)

with open(sys.argv[3], "w") as f:
	json.dump(dclusters, f, indent=4)

