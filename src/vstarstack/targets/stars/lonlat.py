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

import json
import os

import vstarstack.cfg
import vstarstack.usage
import vstarstack.projection.perspective
import vstarstack.common


def process_file(jsonfile):
    with open(jsonfile) as f:
        desc = json.load(f)
    h = desc["h"]
    w = desc["w"]
    proj = desc["projection"]
    if proj == "perspective":
        W = desc["W"]
        H = desc["H"]
        F = desc["F"]
        proj = vstarstack.projection.perspective.Projection(W, H, F, w, h)
    else:
        raise Exception("Unknown projection %s" % proj)

    if "stars" in desc:
        for star in desc["stars"]:
            x = star["x"]
            y = star["y"]
            lat, lon = proj.project(y, x)
            star["lon"] = lon
            star["lat"] = lat
    with open(jsonfile, "w") as f:
        json.dump(desc, f, indent=4)


def process_dir(path):
    descs = vstarstack.common.listfiles(path, ".json")
    for name, filename in descs:
        print(name)
        process_file(filename)


def process(project: vstarstack.cfg.Project, argv: list):
    if len(argv) >= 1:
        path = argv[0]
    else:
        path = project.config["paths"]["descs"]

    if os.path.isdir(path):
        process_dir(path)
    else:
        process_file(path)


def run(project: vstarstack.cfg.Project, argv: list):
    process(project, argv)
