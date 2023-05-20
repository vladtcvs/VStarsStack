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

import vstarstack.tool.cfg

def dircheck(name):
    """Create dir if required"""
    if not os.path.isdir(name):
        os.mkdir(name)

def run(_project: vstarstack.tool.cfg.Project, argv: list):
    if len(argv) > 0:
        path = argv[0]
    else:
        path = os.getcwd()

    projf = os.path.join(path, "project.json")
    with open(projf, encoding='utf8') as f:
        proj = json.load(f)

    proj["planets"] = {
        "map_resolution": 360,
        "paths": {
            "cutted": "cutted",
            "maps": "maps",
        },
    }

    dircheck(path + '/cutted')
    dircheck(path + '/maps')

    with open(projf, "w", encoding='utf8') as f:
        json.dump(proj, f, indent=4, ensure_ascii=False)
