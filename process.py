import planet.planet
import stars.stars
import cluster.cluster

import os

import optical_fix.fixes
import readnef
import merge
import shift

import sys


def usage():
	print("process command ...")
	print("Commands: ")
	print("\treadnef - read nikon NEF to npy")
	print("\toptical-fix - make optical fixes (devignetting, remove distorsion)")
	print("\tplanet - commands for processing planet images")
	print("\tstars - commands for processing stars images")
	print("\tcluster - command for cluster processing")
	print("\tshift - move and rotate images to match them")
	print("\tmerge - merge images")
	print("\thelp - pring help")

def run(argv):
	cmd = argv[0]
	if cmd == "readnef":
		readnef.run(argv[1:])
	elif cmd == "optical-fix":
		optical_fix.fixes.run(argv[1:])
	elif cmd == "planet":
		planet.planet.run(argv[1:])
	elif cmd == "stars":
		stars.stars.run(argv[1:])
	elif cmd == "cluster":
		cluster.cluster.run(argv[1:])
	elif cmd == "shift":
		shift.run(argv[1:])
	elif cmd == "merge":
		merge.run(argv[1:])
	elif cmd == "help":
		usage()
	elif cmd == "stars-process":
		basedir = argv[1]
		if len(argv) > 2:
			result=argv[2]
		else:
			result=os.path.join(basedir, "merged.npz")

		origdir = os.path.join(basedir, "orig")
		npydir = os.path.join(basedir, "npy")
		shifteddir = os.path.join(basedir, "shifted")

		print("Readnef")
		readnef.run((origdir, npydir))
		print("Fix vignetting")
		optical_fix.fixes.run(("vignetting", npydir, npydir))
		print("Fix distorsion")
		optical_fix.fixes.run(("distorsion", npydir, npydir))

		stars.stars.run(("process", basedir))
		print("Apply shifts")
		shift.run((npydir, os.path.join(basedir, "shifts.json"), shifteddir))
		print("Merge")
		merge.run((shifteddir, result))
	else:
		print("Unknown command", cmd)

if __name__ == "__main__":
	run(sys.argv[1:])

