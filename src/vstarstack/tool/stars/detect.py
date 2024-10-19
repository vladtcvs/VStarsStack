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
import vstarstack.library.image_process.togray

def detect_stars(projection,
                 gray : np.ndarray):
    """Detect stars on image"""
    stars = detect.detect_stars(gray)
    for star in stars:
        star["lon"], star["lat"] = projection.project(star["x"], star["y"])
    return sorted(stars, key=lambda x: x["size"], reverse=True)

def get_brightest(stars, N, mindistance):
    """Get first N brightest stars"""
    sample = []
    stars = sorted(stars, key=lambda item: item["size"], reverse=True)
    for star in stars:
        for selected in sample:
            if abs(selected["y"] - star["y"]) < mindistance and \
               abs(selected["x"] - star["x"]) < mindistance:
                break
        else:
            sample.append(star)
            if len(sample) >= N:
                break
    return sample

def _process_file(name, fname, jsonfile, num_stars, mindist):
    """Process single file"""
    image = vstarstack.library.data.DataFrame.load(fname)
    gray,_ = vstarstack.library.image_process.togray.df_to_gray(image)
    projection = vstarstack.library.projection.tools.get_projection(image)
    stars = detect_stars(projection, gray)
    print("Detected ", len(stars))
    stars = get_brightest(stars, num_stars, mindist)
    stars = [{"keypoint" : item} for item in stars]
    desc = {
        "fname" : fname,
        "name" : name,
        "h": image.params["h"],
        "w": image.params["w"],
        "points": stars,
    }

    vstarstack.tool.common.check_dir_exists(jsonfile)
    with open(jsonfile, "w", encoding="utf8") as f:
        json.dump(desc, f, indent=4)

def _process_dir(path, jsonpath, num_stars, mindist):
    files = vstarstack.tool.common.listfiles(path, ".zip")

    for name, filename in files:
        print(name)
        _process_file(name, filename, os.path.join(jsonpath, name + ".json"), num_stars, mindist)

def run(project: vstarstack.tool.cfg.Project, argv: list):
    """Detect stars"""
    if len(argv) >= 2:
        path = argv[0]
        jsonpath = argv[1]
        if len(argv) >= 4:
            num_stars = int(argv[2])
            mindist = float(argv[3])
        else:
            num_stars = -1 # all stars
            mindist = 0.001
    else:
        path = project.config.paths.light.npy
        jsonpath = project.config.paths.descs
        num_stars = project.config.stars.describe.num_main
        mindist = project.config.stars.describe.mindist

    thr_coeff = project.config.stars.brightness_over_neighbours
    detect.configure_detector(thresh_coeff=thr_coeff)
    if os.path.isdir(path):
        _process_dir(path, jsonpath, num_stars, mindist)
    else:
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]
        _process_file(name, path, jsonpath, num_stars, mindist)
