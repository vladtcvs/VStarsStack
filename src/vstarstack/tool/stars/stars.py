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
import vstarstack.tool.stars.show

def _enable_stars(project : vstarstack.tool.cfg.Project, _argv: list[str]):
    project.config.enable_module("stars")
    project.config.enable_module("cluster")
    vstarstack.tool.cfg.store_project()

commands = {
    "config": (_enable_stars, "configure stars pipeline"),
    "detect": (vstarstack.tool.stars.detect.run, "detect stars"),
    "show": (vstarstack.tool.stars.show.show, "display detected stars", "npy/light.zip channel descs/desc.json"),
    "show-match": (vstarstack.tool.stars.show.show_match, "display matched stars", "npy/light1.zip npy/light2.zip descs/desc1.json descs/desc2.json match_table.json"),
    "describe": (vstarstack.tool.stars.describe.run, "find descriptions for each image"),
    "match": (vstarstack.tool.stars.match.run, "match stars between images"),
    "cluster": (vstarstack.tool.stars.build_clusters.run,
                "find matching stars clusters between images"),
}
