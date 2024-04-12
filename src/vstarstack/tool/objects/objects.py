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

import vstarstack.tool.objects.cut
import vstarstack.tool.objects.config
import vstarstack.tool.objects.features

def _enable_objects(project : vstarstack.tool.cfg.Project, _argv: list[str]):
    project.config.enable_module("objects")
    vstarstack.tool.cfg.store_project()

commands = {
    "config": (_enable_objects, "configure compact_objects pipeline"),
    "detect": ("vstarstack.tool.objects.detect", "detect compact objects"),
    "features": (vstarstack.tool.objects.features.run, "detect and match image features"),
    "cut": (vstarstack.tool.objects.cut.run, "cut compact objects"),
}
