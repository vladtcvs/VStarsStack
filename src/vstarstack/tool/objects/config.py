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
    "features" : ("module", {
        "path" : (str, "features/"),
        "num_splits" : (int, 4),
        "max_feature_delta" : (int, 20),
        "features_percent" : (int, 100),
        "bright_spots": {
            "blurSize" : (int, 21),
            "k_thr" : (float, 1.15),
            "minValue" : (float, 0.1),
            "minPixel" : (int, 5),
            "maxPixel" : (int, 20),
        },
        "orb" : {
            "patchSize" : (int, 31),
        },
    }),
}

configuration = Configuration(_module_configuration)
