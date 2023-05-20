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

import vstarstack.tool.planets.buildmap
import vstarstack.tool.planets.configure
import vstarstack.tool.usage
import vstarstack.tool.cfg

commands = {
    "configure": (vstarstack.tool.planets.configure.run, "configure planets in project"),
    "buildmap": (vstarstack.tool.planets.buildmap.run, "build planet surface map"),
}

def run(project: vstarstack.tool.cfg.Project, argv: list):
    vstarstack.tool.usage.run(project, argv, "planets", commands, autohelp=True)
