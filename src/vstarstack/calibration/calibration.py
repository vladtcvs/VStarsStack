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

import vstarstack.calibration.dark
import vstarstack.calibration.flat

import vstarstack.usage
import vstarstack.cfg

commands = {
    "dark": (vstarstack.calibration.dark.run,     "dark frames handling"),
    "flat": (vstarstack.calibration.flat.run,     "flat frames handling"),
}


def run(project: vstarstack.cfg.Project, argv: list):
    vstarstack.usage.run(project, argv, "calibration", commands, autohelp=True)
