import image_fix.flatten
import image_fix.distorsion
import image_fix.remove_sky
import image_fix.difference
import image_fix.border

import os

import common
import cfg
import shutil
import usage

def copy(argv):
	orig = cfg.config["paths"]["npy-orig"]
	fixed = cfg.config["paths"]["npy-fixed"]
	files = common.listfiles(orig, ".zip")
	for name, fname in files:
		print("Copying ", name)
		fname_out = os.path.join(fixed, name + ".zip")
		shutil.copyfile(fname, fname_out)

commands = {
	"copy"       : (copy, "just copy images from original to pipeline dir"),
	"distorsion" : (image_fix.distorsion.run, "fix distorsion"),
	"flatten"    : (image_fix.flatten.run,    "fix vignetting and other similar problems"),
	"remove-sky" : (image_fix.remove_sky.run, "remove sky"),
	"difference" : (image_fix.difference.run, "substract dark frame"),
	"border"     : (image_fix.border.run,     "remove border"),
}

def run(argv):
	usage.run(argv, "image-fix", commands, autohelp=True)

