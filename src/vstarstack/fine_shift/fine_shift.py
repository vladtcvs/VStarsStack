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

import vstarstack.fine_shift.image_wave
import vstarstack.usage
import vstarstack.cfg

import json

Nsteps=10000
dh = 0.1

def align_features(argv):
    images = argv[0]
    clusters = argv[1]
    outpath = argv[2]

    with open(clusters) as f:
        clusters = json.load(f)

    W = vstarstack.cfg.camerad["w"]
    H = vstarstack.cfg.camerad["h"]

    names = []
    good_clusters = []

    for name in names:
        wave = vstarstack.fine_shift.image_wave.ImageWave(W, H, 20, 20)
        points = []
        targets = []
        for cluster in good_clusters:
            x = cluster["average"]["x"]
            y = cluster["average"]["y"]
            targets.append((x,y))
            x = cluster["images"][name]["x"]
            y = cluster["images"][name]["y"]
            points.append((x,y))
        wave.approximate(targets, points, Nsteps, dh)

commands = {
    "align-features" : (align_features, "align features on images", "npy/ clusters.json shifted/"),
}

def run(argv):
    vstarstack.usage.run(argv, "fine-shift", commands, autohelp=True)
