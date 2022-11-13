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


import sys

import vstarstack.targets.compact_objects.compact_objects
import vstarstack.targets.stars.stars
import vstarstack.cluster.cluster
import vstarstack.image_fix.fixes
import vstarstack.calibration.calibration
import vstarstack.readimage.readimage
import vstarstack.merge
import vstarstack.shift.shift
import vstarstack.configurate
import vstarstack.debayer.debayer
import vstarstack.targets.planets.planets

import vstarstack.fine_shift.fine_shift

import vstarstack.image
import vstarstack.clean
import vstarstack.usage

commands = {
	"readimage" : (vstarstack.readimage.readimage.run, "read source images to npz"),
	"debayer"   : (vstarstack.debayer.debayer.run, "debayer RAW images"),
	"image-fix" : (vstarstack.image_fix.fixes.run, "image-fix - make optical fixes (remove distorsion, coma, etc) and other image fixes"),
	"calibration" : (vstarstack.calibration.calibration.run, "calibration - flats, darks"),
	"compact_objects" : (vstarstack.targets.compact_objects.compact_objects.run, "commands for processing images with compact objects (planets, diffractions, etc)"),
	"stars" : (vstarstack.targets.stars.stars.run, "commands for processing stars images"),
	"cluster" : (vstarstack.cluster.cluster.run, "command for cluster processing"),
	"shift" : (vstarstack.shift.shift.run, "move and rotate images to match them"),
	"merge" : (vstarstack.merge.run, "merge images", "input_dir/ output.npz"),
	"project" : (vstarstack.configurate.run, "configurate project"),
	"planets" : (vstarstack.targets.planets.planets.run, "commands for processing planets"),
	"image" : (vstarstack.image.run, "image processing (show, convert, etc)"),
	"clean" : (vstarstack.clean.run, "remove temporary files"),
}

def run(argv, progname=None):
	if progname is not None:
		vstarstack.usage.setprogname(progname)
	vstarstack.usage.run(argv, "", commands, autohelp=True)

if __name__ == "__main__":
	run(sys.argv[2:], sys.argv[1])
