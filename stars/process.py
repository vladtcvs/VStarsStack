import os

import stars.detect
import stars.describe
import stars.match
import stars.net
import stars.cluster
import stars.lonlat
import sys
import usage

def process(argv):
	basedir = argv[0]
	if len(argv) > 1:
		first_command = argv[1]
		if first_command == "detect":
			level = 0
		elif first_command == "lonlat":
			level = 1
		elif first_command == "describe":
			level = 2
		elif first_command == "match":
			level = 3
		elif first_command == "net":
			level = 4
		elif first_command == "cluster":
			level = 5
	else:
		level = 0

	npydir = os.path.join(basedir, "npy")

	starsdir = os.path.join(basedir, "stars")
	if not os.path.exists(starsdir):
		os.mkdir(starsdir)

	descdir = os.path.join(basedir, "descs")	
	if not os.path.exists(descdir):
		os.mkdir(descdir)

	net = os.path.join(basedir, "net.json")
	clusters = os.path.join(basedir, "clusters.json")
	shifts = os.path.join(basedir, "shifts.json")

	if level <= 0:
		print("Detect")
		stars.detect.run([npydir, starsdir])
	if level <= 1:
		print("Lonlat")
		stars.lonlat.run([starsdir])
	if level <= 2:
		print("Describe")
		stars.describe.run([starsdir, descdir])
	if level <= 3:
		print("Match")
		stars.match.run([descdir])
	if level <= 4:
		print("Net")
		stars.net.run([descdir, net])
	if level <= 5:
		print("Cluster")
		stars.cluster.run([net, descdir, clusters])

commands = {
	"*" : (process, "Process stars: detect, lonlat, describe, match, net, cluster", "base_dir/ [first_command]"),
}

def run(argv):
	usage.run(argv, "stars process", commands)

