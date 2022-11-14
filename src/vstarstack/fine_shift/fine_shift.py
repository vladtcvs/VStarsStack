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

import os
import json

def cluster_average(cluster):
    xs = []
    ys = []
    for name in cluster:
        xs.append(cluster[name]["x"])
        ys.append(cluster[name]["y"])
    pos = {
        "x" : sum(xs)/len(xs),
        "y" : sum(ys)/len(ys),
    }
    return pos

def find_alignment(argv):
    clusters = argv[0]
    outpath = argv[1]

    with open(clusters) as f:
        clusters = json.load(f)

    W = vstarstack.cfg.camerad["w"]
    H = vstarstack.cfg.camerad["h"]

    Nsteps = vstarstack.cfg.config["fine_shift"]["Nsteps"]
    dh = vstarstack.cfg.config["fine_shift"]["dh"]
    gridW = vstarstack.cfg.config["fine_shift"]["gridW"]
    gridH = vstarstack.cfg.config["fine_shift"]["gridH"]
    

    names = []
    good_clusters = []
    for cluster in clusters:
        names += list(cluster.keys())
        gcluster = {
            "average" : cluster_average(cluster),
            "images" : cluster,
        }
        good_clusters.append(gcluster)
    names = sorted(list(set(names)))

    print("Names: ", names)

    for name in names:
        print("Processing: %s" % name)
        wave = vstarstack.fine_shift.image_wave.ImageWave(W, H, gridW, gridH)
        points = []
        targets = []
        for cluster in good_clusters:
            if name not in cluster["images"]:
                continue
            x = cluster["average"]["x"]
            y = cluster["average"]["y"]
            targets.append((x,y))
            x = cluster["images"][name]["x"]
            y = cluster["images"][name]["y"]
            points.append((x,y))
        print("\tusing %i points" % len(points))
        wave.approximate(targets, points, Nsteps, dh)
        data = wave.data()
        with open(os.path.join(outpath, name+".json"), "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

commands = {
    "align-features" : (find_alignment, "find alignment of images", "clusters.json alignments/"),
}

def run(argv):
    vstarstack.usage.run(argv, "fine-shift", commands, autohelp=True)
