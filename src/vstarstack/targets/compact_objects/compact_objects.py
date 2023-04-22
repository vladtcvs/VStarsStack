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

import vstarstack.targets.compact_objects.detect
import vstarstack.targets.compact_objects.cut
import vstarstack.targets.compact_objects.configure
import vstarstack.targets.compact_objects.features

import vstarstack.usage

commands = {
    "config": (vstarstack.targets.compact_objects.configure.run, "configure compact_objects pipeline"),
    "detect": (vstarstack.targets.compact_objects.detect.run, "detect compact objects"),
    "cut": (vstarstack.targets.compact_objects.cut.run, "cut compact objects"),
    "features": (vstarstack.targets.compact_objects.features.run, "detect and match features on images")
}


def run(project: vstarstack.cfg.Project, argv: list):
    vstarstack.usage.run(argv, "compact_objects", commands, autohelp=True)
