#
# Copyright (c) 2022-2024 Vladislav Tsendrovskii
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

import vstarstack.tool.objects.cut
import vstarstack.tool.objects.config
import vstarstack.tool.objects.find_features
import vstarstack.tool.objects.match_features
import vstarstack.tool.objects.show

def _enable_objects(project : vstarstack.tool.cfg.Project, _argv: list[str]):
    project.config.enable_module("objects")
    vstarstack.tool.cfg.store_project()

commands = {
    "config": (_enable_objects, "configure compact_objects pipeline"),
    "detect": ("vstarstack.tool.objects.detect", "detect compact objects"),
    "show" : (vstarstack.tool.objects.show.run, "display detected compact objects", "image.zip desc.json"),
    "find-features": (vstarstack.tool.objects.find_features.commands, "detect image features"),
    "display-features" : (vstarstack.tool.objects.find_features.display_features, "display image features", "image.zip features.json"),
    "match-features": (vstarstack.tool.objects.match_features.run, "match image features", "features/ match_table.json [comparsion_list.csv]"),
    "cut": (vstarstack.tool.objects.cut.run, "cut compact objects"),
}
