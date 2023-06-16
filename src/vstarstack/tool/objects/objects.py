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

import vstarstack.tool.objects.detect
import vstarstack.tool.objects.cut
import vstarstack.tool.objects.configure
import vstarstack.tool.objects.clusters

commands = {
    "config": (vstarstack.tool.objects.configure.run, "configure compact_objects pipeline"),
    "detect": (vstarstack.tool.objects.detect.run, "detect compact objects"),
    "cut": (vstarstack.tool.objects.cut.run, "cut compact objects"),
    "clusters": (vstarstack.tool.objects.clusters.run, "detect and match features on images")
}

def run(project: vstarstack.tool.cfg.Project, argv: list[str]):
    vstarstack.tool.usage.run(project, argv, "compact_objects", commands, autohelp=True)
