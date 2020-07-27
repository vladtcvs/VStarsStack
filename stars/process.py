import os

import stars.detect
import stars.describe
import stars.match
import stars.net
import stars.cluster
import stars.find_shift
import stars.lonlat
import sys

def run(argv):
	basedir = argv[0]
	npydir = os.path.join(basedir, "npy")
	starsdir = os.path.join(basedir, "stars")
	descdir = os.path.join(basedir, "descs")
	
	net = os.path.join(basedir, "net.json")
	clusters = os.path.join(basedir, "clusters.json")
	shifts = os.path.join(basedir, "shifts.json")

	print("Detect")
	stars.detect.run([npydir, starsdir])
	print("Lonlat")
	stars.lonlat.run([starsdir])
	print("Describe")
	stars.describe.run([starsdir, descdir])
	print("Match")
	stars.match.run([descdir])
	print("Net")
	stars.net.run([descdir, net])
	print("Cluster")
	stars.cluster.run([net, descdir, clusters])
	print("Find shift")
	stars.find_shift.run([clusters, shifts])

if __name__ == "__main__":
	run(sys.argv[1:])

