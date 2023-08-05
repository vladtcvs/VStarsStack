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
import vstarstack.tool.configuration
import vstarstack.tool.fine_shift.align

def _enable_fine_shift(project : vstarstack.tool.cfg.Project, _argv: list[str]):
    project.config.enable_module("fine_shift")
    vstarstack.tool.cfg.store_project()

commands = {
    "config": (_enable_fine_shift, "configure fine_shift pipeline"),
    "align": (vstarstack.tool.fine_shift.align.align, "align images"),
}
