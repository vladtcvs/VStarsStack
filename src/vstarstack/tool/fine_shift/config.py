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

import vstarstack.tool.configuration

_module_configuration = {
    "Nsteps" : (int, 2000),
    "dh" : (float, 0.1),
    "gridW" : (int, 64),
    "gridH" : (int, 64),
    "stretchPenaltyCoefficient" : (float, 0.01),
    "points_min_len" : (int, 0),
    "aligns" : (str, "aligns/"),
    "subpixels" : (int, 1),
    "max_shift" : (int, 15),
    "correlation_grid" : (int, 15),
}

configuration = vstarstack.tool.configuration.Configuration(_module_configuration)
