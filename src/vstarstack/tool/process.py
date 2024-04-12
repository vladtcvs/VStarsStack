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


import vstarstack.tool.cfg
import vstarstack.tool.usage
import vstarstack.tool.manage_project

import vstarstack.tool.clean

commands = {
    "readimage": ("vstarstack.tool.readimage",
                  "read source images to npz"),
    "debayer": ("vstarstack.tool.debayer",
                "debayer RAW images"),
    "process": ("vstarstack.tool.image_processing.fixes",
                "process - make optical fixes and other image fixes"),
    "calibration": ("vstarstack.tool.calibration",
                    "calibration - flats, darks"),
    "objects": ("vstarstack.tool.objects.objects",
                "commands for processing images with compact objects " +
                "(planets, diffractions, etc)"),
    "stars": ("vstarstack.tool.stars.stars",
              "commands for processing stars images"),
    "cluster": ("vstarstack.tool.cluster",
                "command for cluster processing"),
    "shift": ("vstarstack.tool.shift",
              "move and rotate images to match them"),
    "merge": ("vstarstack.tool.merge",
              "merge images", "input_dir/ output.npz"),
    "project": (vstarstack.tool.manage_project.run,
                "configurate project"),
#    "planets": (vstarstack.targets.planets.planets.run,
#                "commands for processing planets"),
    "image": ("vstarstack.tool.image",
              "image processing (show, convert, etc)"),
    "clean": (vstarstack.tool.clean.run,
              "remove temporary files"),
    "fine-shift": ("vstarstack.tool.fine_shift.fine_shift",
                   "fine shift images"),
    "analyzers": ("vstarstack.tool.analyzers.analyzers",
                   "analyze images"),
}
