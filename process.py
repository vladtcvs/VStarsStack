import planet.planet
import stars.stars
import cluster.cluster

import os

import image_fix.fixes
import readimage.readimage
import merge
import shift
import configurate

import sys
import usage

commands = {
	"readimage" : (readimage.readimage.run, "read source images to npz"),
	"image-fix" : (image_fix.fixes.run, "image-fix - make optical fixes (devignetting, remove distorsion) and other image fixes"),
	"planets" : (planet.planet.run, "commands for processing planet images"),
	"stars" : (stars.stars.run, "commands for processing stars images"),
	"cluster" : (cluster.cluster.run, "command for cluster processing"),
	"shift" : (shift.run, "move and rotate images to match them"),
	"merge" : (merge.run, "merge images", "input_dir/ output.npz"),
	"configurate" : (configurate.run, "configurate project"),
}

def run(argv):
	usage.run(argv, "", commands)

if __name__ == "__main__":
	run(sys.argv[1:])

