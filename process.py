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


import targets.compact_objects.compact_objects
import targets.stars.stars
import cluster.cluster

import os

import image_fix.fixes
import calibration.calibration
import readimage.readimage
import merge
import shift.shift
import configurate
import debayer.debayer
import targets.planets.planets

import image
import clean

import sys
import usage

commands = {
	"readimage" : (readimage.readimage.run, "read source images to npz"),
	"debayer"   : (debayer.debayer.run, "debayer RAW images"),
	"image-fix" : (image_fix.fixes.run, "image-fix - make optical fixes (remove distorsion, coma, etc) and other image fixes"),
	"calibration" : (calibration.calibration.run, "calibration - flats, darks"),
	"compact_objects" : (targets.compact_objects.compact_objects.run, "commands for processing images with compact objects (planets, diffractions, etc)"),
	"stars" : (targets.stars.stars.run, "commands for processing stars images"),
	"cluster" : (cluster.cluster.run, "command for cluster processing"),
	"shift" : (shift.shift.run, "move and rotate images to match them"),
	"merge" : (merge.run, "merge images", "input_dir/ output.npz"),
	"project" : (configurate.run, "configurate project"),
	"planets" : (targets.planets.planets.run, "commands for processing planets"),
	"image" : (image.run, "image processing (show, convert, etc)"),
	"clean" : (clean.run, "remove temporary files"),
}

def run(argv, progname=None):
	if progname is not None:
		usage.setprogname(progname)
	usage.run(argv, "", commands, autohelp=True)

if __name__ == "__main__":
	run(sys.argv[2:], sys.argv[1])

