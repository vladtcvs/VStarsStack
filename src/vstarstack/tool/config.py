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
    "paths": {
        "original": (str, "orig", {"directory" : True}),
        "npy_orig": (str, "npy_orig", {"directory" : True}),
        "npy_fixed": (str, "npy", {"directory" : True}),
        "aligned": (str, "aligned", {"directory" : True}),
        "output": (str, "sum.zip"),
        "descs" : (str, "descs", {"directory" : True}),
        "relative_shifts": (str, "shifts.json"),
        "absolute_shifts": (str, "shift.json"),
        "shift_errors" : (str, "errors.csv"),
    },
    "telescope": {
        "camera": {
            "pixel_W": (float, 3.0),
            "pixel_H": (float, 3.0),
            "format": (str, "flat"),
        },
        "scope": {
            "F": (float, 1000.0),
            "D": (float, 100.0),
        },
    },
    "merge" : {
        "sigma_clip_coefficient_begin" : (float, 4.0),
        "sigma_clip_coefficient_end" : (float, 2.0),
        "sigma_clip_steps" : (int, 2),
    },
    "use_modules" : (list, []),
    "cluster" : ("module", {
        "path" : (str, "clusters.json"),
        "compose_movements" : (bool, True),
    }),
    "stars" : ("module", vstarstack.tool.stars.config.configuration),
    "objects" : ("module", vstarstack.tool.objects.config.configuration),
    "planets" : ("module", vstarstack.tool.planets.config.configuration),
    "fine_shift" : ("module", vstarstack.tool.fine_shift.config.configuration),
}

configuration = Configuration(_module_configuration)
