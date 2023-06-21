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

import vstarstack.tool.cfg
import vstarstack.tool.usage
import vstarstack.tool.manage_project

import vstarstack.tool.config
import vstarstack.tool.image
import vstarstack.tool.clean
import vstarstack.tool.readimage
import vstarstack.tool.merge
import vstarstack.tool.debayer
import vstarstack.tool.stars.stars
import vstarstack.tool.cluster
import vstarstack.tool.shift
import vstarstack.tool.image_processing.fixes
import vstarstack.tool.fine_shift
import vstarstack.tool.objects.objects
import vstarstack.tool.calibration

commands = {
    "readimage": (vstarstack.tool.readimage.run,
                  "read source images to npz"),
    "debayer": (vstarstack.tool.debayer.run,
                "debayer RAW images"),
    "image-process": (vstarstack.tool.image_processing.fixes.run,
                  "image-process - make optical fixes and other image fixes"),
    "calibration": (vstarstack.tool.calibration.run,
                    "calibration - flats, darks"),
    "objects": (vstarstack.tool.objects.objects.run,
                "commands for processing images with compact objects (planets, diffractions, etc)"),
    "stars": (vstarstack.tool.stars.stars.run,
              "commands for processing stars images"),
    "cluster": (vstarstack.tool.cluster.run,
                "command for cluster processing"),
    "shift": (vstarstack.tool.shift.run,
              "move and rotate images to match them"),
    "merge": (vstarstack.tool.merge.run,
              "merge images", "input_dir/ output.npz"),
    "project": (vstarstack.tool.manage_project.run,
                "configurate project"),
#    "planets": (vstarstack.targets.planets.planets.run,
#                "commands for processing planets"),
    "image": (vstarstack.tool.image.run,
              "image processing (show, convert, etc)"),
    "clean": (vstarstack.tool.clean.run,
              "remove temporary files"),
    "fine-shift": (vstarstack.tool.fine_shift.run,
                   "fine shift images"),
}


def run(project: vstarstack.tool.cfg.Project, argv: list, progname=None):
    """Run program"""
    if progname is not None:
        vstarstack.tool.usage.setprogname(progname)
    vstarstack.tool.usage.run(project, argv, "", commands, autohelp=True)


if __name__ == "__main__":
    program_project = vstarstack.tool.cfg.get_project()
    program_argv = [item for item in sys.argv[2:] if item[:2] != "--"]
    run(program_project, program_argv, sys.argv[1])
