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

import os
import json

import vstarstack.cfg

def dircheck(name):
    if not os.path.isdir(name):
        os.mkdir(name)


def run(_project: vstarstack.cfg.Project, argv: list):
    if len(argv) > 0:
        dir = argv[0]
    else:
        dir = os.getcwd()

    projf = os.path.join(dir, "project.json")
    with open(projf) as f:
        proj = json.load(f)

    proj["mode"] = "compact_objects"
    proj["compact_objects"] = {
        "threshold": 0.05,
        "disc": {
            "mindelta": 40,
            "maxdelta": 50,
            "num_bins_curvature": 50,
            "num_bins_distance": 10,
        },
        "brightness": {
            "min_diameter": 20,
            "max_diameter": 40,
        },
        "margin": 20,
        "require_size": True,
    }

    dircheck(dir + '/descs')

    with open(projf, "w") as f:
        json.dump(proj, f, indent=4, ensure_ascii=False)
