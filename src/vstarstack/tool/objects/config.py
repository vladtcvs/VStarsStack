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

from vstarstack.tool.configuration import Configuration

_module_configuration = {
    "threshold": (float, 0.05),
    "margin": (int, 30),
    "require_size": (bool, True),
    "use_modules" : (list, ["disc", "brightness"]),
    "disc": ("module", {
        "mindelta": (int, 40),
        "maxdelta": (int, 50),
        "num_bins_curvature": (int, 50),
        "num_bins_distance": (int, 10),
    }),
    "brightness": ("module", {
        "min_diameter": (int, 20),
        "max_diameter": (int, 40),
    }),
}

configuration = Configuration(_module_configuration)
