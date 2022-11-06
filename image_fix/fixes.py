#
# Copyright (c) 2022 Vladislav Tsendrovskii
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

import image_fix.distorsion
import image_fix.remove_sky
import image_fix.border
import image_fix.normalize
import image_fix.motion_fix
import image_fix.deconvolution
import image_fix.useL

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
	"normalize"  : (image_fix.normalize.run,  "normalize to weight"),
	"fix-motion" : (image_fix.motion_fix.run, "remove motion of image"),
	"deconvolution" : (image_fix.deconvolution.run, "deconvolution of image"),
	"useL"       : (image_fix.useL.run, "use L channel for brightness"),
}

def run(argv):
	usage.run(argv, "image-fix", commands, autohelp=True)
