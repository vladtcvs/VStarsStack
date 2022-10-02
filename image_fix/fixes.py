import image_fix.distorsion
import image_fix.remove_sky
import image_fix.border
import image_fix.normalize

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
	"remove-sky" : (image_fix.remove_sky.run, "remove sky"),
	"border"     : (image_fix.border.run,     "remove border"),
	"normalize"     : (image_fix.normalize.run,     "normalize to weight"),
}

def run(argv):
	usage.run(argv, "image-fix", commands, autohelp=True)

