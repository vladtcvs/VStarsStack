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

import vstarstack.tool.stars.config
import vstarstack.tool.objects.config
import vstarstack.tool.planets.config
import vstarstack.tool.fine_shift.config

_module_configuration = {
    "output" : {
        "projection_type" : (str, "COPY"),
        "projection_desc" : (str, "COPY"),
    },
    "paths": {
        "light" : {
            "original": (str, "orig/light", {"directory" : True}),
            "npy": (str, "npy/light", {"directory" : True}),
            "result" : (str, "light.zip"),
        },
        "flat" : {
            "original": (str, "orig/flat", {"directory" : True}),
            "npy": (str, "npy/flat", {"directory" : True}),
            "result" : (str, "flat.zip"),
        },
        "dark" : {
            "original": (str, "orig/dark", {"directory" : True}),
            "npy": (str, "npy/dark", {"directory" : True}),
            "result" : (str, "dark.zip"),
        },
        "aligned": (str, "aligned", {"directory" : True}),
        "descs" : (str, "descs", {"directory" : True}),
        "relative_shifts": (str, "shifts.json"),
        "absolute_shifts": (str, "shift.json"),
        "shift_errors" : (str, "errors.csv"),
        "photometry" : (str, "photometry", {"directory" : True}),
    },
    "telescope": {
        "camera": {
            "pixel_W": (float, 3.0),
            "pixel_H": (float, 3.0),
            "format": (str, "COPY"),
        },
        "scope": {
            "F": (float, 1000.0),
            "D": (float, 100.0),
        },
    },
    "shift" : {
        "interpolate" : (bool, None),
    },
    "darks" : {
        "basic_temperature" : (float, -10),
        "delta_temperature" : (float, 2),
    },
    "merge" : {
        "sigma_clip_coefficient_begin" : (float, 4.0),
        "sigma_clip_coefficient_end" : (float, 2.0),
        "sigma_clip_steps" : (int, 2),
    },
    "use_modules" : (list, ["cluster"]),
    "cluster" : ("module", {
        "path" : (str, "clusters.json"),
        "matchtable" : (str, "match_table.json"),
        "compose_movements" : (bool, True),
    }),
    "stars" : ("module", vstarstack.tool.stars.config.configuration),
    "objects" : ("module", vstarstack.tool.objects.config.configuration),
    "planets" : ("module", vstarstack.tool.planets.config.configuration),
    "fine_shift" : ("module", vstarstack.tool.fine_shift.config.configuration),
}

configuration = Configuration(_module_configuration)
