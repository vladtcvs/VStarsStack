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
import cv2

import vstarstack.data
import sys
import json
import os

import vstarstack.cfg
import vstarstack.common
import vstarstack.usage

import vstarstack.targets.compact_objects.detectors.brightness_detector as bd
import vstarstack.targets.compact_objects.detectors.disc_detector as dd


def process_file(filename, descfilename, detector):
    image = vstarstack.data.DataFrame.load(filename)

    for channel in image.get_channels():
        layer, opts = image.get_channel(channel)
        if not opts["brightness"]:
            continue
        layer = layer / np.amax(layer)

        planet = detector(layer, debug=vstarstack.cfg.debug)

        if planet is not None:
            break
    else:
        print("No planet detected")
        return

    desc = {
        "compact_object": planet,
    }

    with open(descfilename, "w") as f:
        json.dump(desc, f, indent=4)


def process_path(npys, descs, detector):
    files = vstarstack.common.listfiles(npys, ".zip")
    for name, filename in files:
        print(name)
        out = os.path.join(descs, name + ".json")
        process_file(filename, out, detector)


def process(project, detector, argv):
    if len(argv) > 0:
        input = argv[0]
        output = argv[1]
        if os.path.isdir(input):
            process_path(input, output, detector)
        else:
            process_file(input, output, detector)
    else:
        process_path(project.config["paths"]["npy-fixed"],
                     project.config["paths"]["descs"], detector)


def process_brightness(argv):
    process(bd.detect, argv)


def process_disc(argv):
    process(dd.detect, argv)


commands = {
    "brightness": (process_brightness, "detect compact objects with brightness detector", "npy/ descs/"),
    "disc": (process_disc, "detect compact objects with disc detector", "npy/ descs/"),
}


def run(project: vstarstack.cfg.Project, argv: list):
    vstarstack.usage.run(argv, "compact_objects detect",
                         commands, autohelp=True)
