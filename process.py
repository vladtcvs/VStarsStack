import compact_objects.compact_objects
import stars.stars
import cluster.cluster

import os

import image_fix.fixes
import calibration.calibration
import readimage.readimage
import merge
import shift.shift
import configurate
import debayer.debayer
import planets.planets

import sys
import usage

commands = {
	"readimage" : (readimage.readimage.run, "read source images to npz"),
	"debayer"   : (debayer.debayer.run, "debayer RAW images"),
	"image-fix" : (image_fix.fixes.run, "image-fix - make optical fixes (remove distorsion, coma, etc) and other image fixes"),
	"calibration" : (calibration.calibration.run, "calibration - flats, darks"),
	"compact_objects" : (compact_objects.compact_objects.run, "commands for processing images with compact objects (planets, diffractions, etc)"),
	"stars" : (stars.stars.run, "commands for processing stars images"),
	"cluster" : (cluster.cluster.run, "command for cluster processing"),
	"shift" : (shift.shift.run, "move and rotate images to match them"),
	"merge" : (merge.run, "merge images", "input_dir/ output.npz"),
	"project" : (configurate.run, "configurate project"),
	"planets" : (planets.planets.run, "commands for processing planets"),
}

def run(argv, progname=None):
	if progname is not None:
		usage.setprogname(progname)
	usage.run(argv, "", commands, autohelp=True)

if __name__ == "__main__":
	run(sys.argv[2:], sys.argv[1])

