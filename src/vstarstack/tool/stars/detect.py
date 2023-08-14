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
import numpy  as np

import vstarstack.tool.common
import vstarstack.tool.cfg
import vstarstack.library.data
import vstarstack.library.common
import vstarstack.library.projection.tools
from vstarstack.library.stars import detect

def detect_stars(projection,
                 gray : np.ndarray):
    """Detect stars on image"""
    stars = detect.detect_stars(gray)
    for star in stars:
        star["lon"], star["lat"] = projection.project(star["x"], star["y"])
    return sorted(stars, key=lambda x: x["size"], reverse=True)

def _process_file(fname, jsonfile):
    """Process single file"""
    image = vstarstack.library.data.DataFrame.load(fname)

    sources = []
    for channel in image.get_channels():
        layer, options = image.get_channel(channel)
        if not options["brightness"]:
            continue
        layer = layer / np.amax(layer)
        sources.append(layer)
    if len(sources) == 0:
        return
    gray = sum(sources)

    projection = vstarstack.library.projection.tools.get_projection(image)
    stars = detect_stars(projection, gray)
    desc = {
        "stars": stars,
        "h": image.params["h"],
        "w": image.params["w"],
    }

    vstarstack.tool.common.check_dir_exists(jsonfile)
    with open(jsonfile, "w", encoding="utf8") as f:
        json.dump(desc, f, indent=4)

def _process_dir(path, jsonpath):
    files = vstarstack.tool.common.listfiles(path, ".zip")

    for name, filename in files:
        print(name)
        _process_file(filename, os.path.join(jsonpath, name + ".json"))

def run(project: vstarstack.tool.cfg.Project, argv: list):
    """Detect stars"""
    if len(argv) >= 2:
        path = argv[0]
        jsonpath = argv[1]
    else:
        path = project.config.paths.npy_fixed
        jsonpath = project.config.paths.descs

    thr_coeff = project.config.stars.brightness_over_neighbours
    detect.configure_detector(thresh_coeff=thr_coeff)
    if os.path.isdir(path):
        _process_dir(path, jsonpath)
    else:
        _process_file(path, jsonpath)
