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

from vstarstack.tool.configuration import Configuration

_module_configuration = {
    "describe" : {
        "num_main" : (int, 15),
        "mindist" : (float, 0.05),
    },
    "match" : {
        "max_angle_diff_k": (float, 0.005),
        "max_size_diff": (float, 0.2),
        "max_dangle_diff": (float, 1.5),
        "min_matched_ditems": (int, 20),
    },
    "use_angles": (bool, True),
    "brightness_over_neighbours": (float, 2.0),
    "max_compares": (int, 0),
}

configuration = Configuration(_module_configuration)
