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

import os
import json

import vstarstack.tool.cfg
import vstarstack.tool.usage

def run(_project: vstarstack.tool.cfg.Project, argv: list):
    """Configure 'stars' section in project"""
    if len(argv) > 0:
        path = argv[0]
    else:
        path = os.getcwd()

    projf = os.path.join(path, "project.json")
    with open(projf, encoding="utf8") as file:
        proj = json.load(file)

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
        "use_angles": True,
        "brightness_over_neighbours": 0.04,
        "paths" : {
            "descs" : "descs",
            "matchfile" : "stars_match.json",
        },
    }
    proj["cluster"] = {
        "path": "clusters.json"
    }

    with open(projf, "w", encoding="utf8") as file:
        json.dump(proj, file, indent=4, ensure_ascii=False)
