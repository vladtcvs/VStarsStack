#
# Copyright (c) 2023 Vladislav Tsendrovskii
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

import vstarstack.tool.stars.config
import vstarstack.tool.stars.detect
import vstarstack.tool.stars.describe
import vstarstack.tool.stars.match
import vstarstack.tool.stars.build_clusters

def _enable_stars(project : vstarstack.tool.cfg.Project, _argv: list[str]):
    project.config.enable_module("stars")
    project.config.enable_module("cluster")
    vstarstack.tool.cfg.store_project()

commands = {
    "config": (_enable_stars, "configure stars pipeline"),
    "detect": (vstarstack.tool.stars.detect.run, "detect stars"),
    "describe": (vstarstack.tool.stars.describe.run, "find descriptions for each image"),
    "match": (vstarstack.tool.stars.match.run, "match stars between images"),
    "cluster": (vstarstack.tool.stars.build_clusters.run, "find matching stars clusters between images"),
}


def run(project: vstarstack.tool.cfg.Project, argv: list):
    vstarstack.tool.usage.run(project, argv, "stars", commands, autohelp=True)
