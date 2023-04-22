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

import numpy as np

import json
import os

import vstarstack.usage

import vstarstack.common
import vstarstack.data
import vstarstack.cfg

import vstarstack.targets.stars.detector.detector

detect = vstarstack.targets.stars.detector.detector.detect_stars


def process_file(project, fname, jsonfile):
    image = vstarstack.data.DataFrame.load(fname)

    sources = []
    for channel in image.get_channels():
        layer, options = image.get_channel(channel)
        if not options["brightness"]:
            continue
        layer = layer / np.amax(layer)
        sources.append(layer)
    gray = sum(sources)

    stars = detect(project, gray, debug=False)[0]
    desc = {
        "stars": stars,
        "h": image.params["h"],
        "w": image.params["w"],
        "projection": image.params["projection"],
        "H": image.params["perspective_kh"] * image.params["h"],
        "W": image.params["perspective_kw"] * image.params["w"],
        "F": image.params["perspective_F"],
    }

    with open(jsonfile, "w") as f:
        json.dump(desc, f, indent=4)


def process_dir(project, path, jsonpath):
    files = vstarstack.common.listfiles(path, ".zip")

    for name, filename in files:
        print(name)
        process_file(project, filename, os.path.join(jsonpath, name + ".json"))


def process(project: vstarstack.cfg.Project, argv: list):
    if len(argv) >= 2:
        path = argv[0]
        jsonpath = argv[1]
    else:
        path = project.config["paths"]["npy-fixed"]
        jsonpath = project.config["paths"]["descs"]

    if os.path.isdir(path):
        process_dir(project, path, jsonpath)
    else:
        process_file(project, path, jsonpath)


def run(project: vstarstack.cfg.Project, argv: list):
    process(project, argv)
