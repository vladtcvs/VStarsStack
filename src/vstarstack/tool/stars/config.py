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
        "num_main" : (int, 20),
        "mindist" : (float, 0.1),
    },
    "match" : {
        "max_angle_diff_k": (float, 0.005),
        "max_size_diff": (float, 0.1),
        "max_dangle_diff": (float, 2.0),
        "min_matched_ditems": (int, 15),
    },
    "paths" : {
        "matchfile" : (str, "match_table.json"),
    },
    "use_angles": (bool, True),
    "brightness_over_neighbours": (float, 0.04),
}

configuration = Configuration(_module_configuration)
