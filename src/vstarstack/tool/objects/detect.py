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
import numpy as np

import vstarstack.library.data
import vstarstack.tool.cfg
import vstarstack.library.common
import vstarstack.tool.usage

import vstarstack.library.objects.brightness_detector as bd
import vstarstack.library.objects.disc_detector as dd

def _process_file(project, filename, descfilename, detector):
    image = vstarstack.library.data.DataFrame.load(filename)

    if os.path.isfile(descfilename):
        with open(descfilename, encoding='utf8') as f:
            desc = json.load(f)
    else:
        desc = {}
    desc["object"] = {}

    gray = None

    for channel in image.get_channels():
        layer, opts = image.get_channel(channel)
        if not opts["brightness"]:
            continue

        if gray is None:
            gray = layer
        else:
            gray += layer

    if gray is None:
        return

    gray = gray / np.amax(gray)

    thresh = project.config.objects.threshold
    if detector == "disc":
        mindelta = project.config.objects.disc.mindelta
        maxdelta = project.config.objects.disc.maxdelta
        num_bin_curv = project.config.objects.disc.num_bins_curvature
        num_bin_dist = project.config.objects.disc.num_bins_distance
        planets = dd.detect(gray, thresh, mindelta, maxdelta, num_bin_curv, num_bin_dist)
    elif detector == "brightness":
        min_size = project.config.objects.brightness.min_diameter
        max_size = project.config.objects.brightness.max_diameter
        planets = bd.detect(gray, min_size, max_size, thresh)
    else:
        raise Exception(f"Invalid detector {detector}")

    if len(planets) > 0:
        desc["object"] = planets[0]
        with open(descfilename, "w", encoding='utf8') as f:
            json.dump(desc, f, indent=4)

def _process_path(project, npys, descs, detector):
    files = vstarstack.library.common.listfiles(npys, ".zip")
    for name, filename in files:
        print(name)
        out = os.path.join(descs, name + ".json")
        _process_file(project, filename, out, detector)

def _process(project, detector, argv):
    if len(argv) > 0:
        input_path = argv[0]
        output_path = argv[1]
        if os.path.isdir(input_path):
            _process_path(project, input_path, output_path, detector)
        else:
            _process_file(project, input_path, output_path, detector)
    else:
        _process_path(project,
                      project.config.paths.npy_fixed,
                      project.config.paths.descs,
                      detector)

def _process_brightness(project, argv):
    _process(project, "brightness", argv)

def _process_disc(project, argv):
    _process(project, "disc", argv)

commands = {
    "brightness": (_process_brightness,
                   "detect compact objects with brightness detector",
                   "npy/ descs/"),
    "disc": (_process_disc,
             "detect compact objects with disc detector",
             "npy/ descs/"),
}
