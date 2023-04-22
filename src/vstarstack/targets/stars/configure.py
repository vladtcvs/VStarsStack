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


def run(project: vstarstack.cfg.Project, argv: list):
    if len(argv) > 0:
        dir = argv[0]
    else:
        dir = os.getcwd()

    projf = os.path.join(dir, "project.json")
    with open(projf) as f:
        proj = json.load(f)

    proj["mode"] = "stars"
    proj["stars"] = {
        "describe": {
            "num_main": 20,
            "mindist": 0.1,
        },
        "match": {
            "max_angle_diff": 0.01,
            "max_size_diff": 0.1,
            "max_dangle_diff": 4,
            "min_matched_ditems": 15,
        },
        "paths": {
            "net": "net.json",
        },
        "use_angles": True,
        "brightness_over_neighbours": 0.04,
    }
    proj["cluster"] = {
        "path": "clusters.json"
    }

    with open(projf, "w") as f:
        json.dump(proj, f, indent=4, ensure_ascii=False)
