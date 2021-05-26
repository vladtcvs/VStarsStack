import planet.planet
import stars.stars
import cluster.cluster

import os

import image_fix.fixes
import readnef
import merge
import shift

import sys

commands = {
	"readnef" : (readnef.run, "read nikon NEF to npz"),
	"image-fix" : (image_fix.fixes.run, "image-fix - make optical fixes (devignetting, remove distorsion) and other image fixes"),
	"planets" : (planet.planet.run, "commands for processing planet images"),
	"stars" : (stars.stars.run, "commands for processing stars images"),
	"cluster" : (cluster.cluster.run, "command for cluster processing"),
	"shift" : (shift.run, "move and rotate images to match them"),
	"merge" : (merge.run, "merge images"),
}

def usage():
	print("process command ...")
	print("Commands: ")
	for cmd in commands:
		print("\t%s - %s" % (cmd, commands[cmd][1]))
	print("\thelp - pring help")

def run(argv):
	cmd = argv[0]
	if cmd not in commands:
		usage()
		return
	commands[cmd][0](argv[1:])

if __name__ == "__main__":
	run(sys.argv[1:])

