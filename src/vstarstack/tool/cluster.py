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
import numpy as np
import cv2
import matplotlib.pyplot as plt

import vstarstack.tool.cfg
import vstarstack.tool.usage

import vstarstack.library.data
from vstarstack.library.movement.find_shift import build_movements
from vstarstack.library.movement.sphere import Movement

def display(_project: vstarstack.tool.cfg.Project, argv: list):
    """Display clusters"""
    slope = vstarstack.tool.cfg.get_param("multiply", float, 1)
    with open(argv[0], encoding='utf8') as file:
        clusters = json.load(file)
    channel = argv[1]
    fname1 = argv[2]
    fname2 = argv[3]

    df1 = vstarstack.library.data.DataFrame.load(fname1)
    df2 = vstarstack.library.data.DataFrame.load(fname2)

    img1, _ = df1.get_channel(channel)
    img2, _ = df2.get_channel(channel)


    img1 = np.clip(img1/np.amax(img1)*slope, 0, 1)
    img2 = np.clip(img2/np.amax(img2)*slope, 0, 1)

    img1 = (img1 * 255).astype(np.uint8)
    img2 = (img2 * 255).astype(np.uint8)

    name1 = os.path.splitext(os.path.basename(fname1))[0]
    name2 = os.path.splitext(os.path.basename(fname2))[0]
    used_clusters = []
    for cluster in clusters:
        if name1 not in cluster or name2 not in cluster:
            continue
        used_cluster = {
            name1: cluster[name1],
            name2: cluster[name2],
        }
        used_clusters.append(used_cluster)

    kps1 = []
    kps2 = []
    matches = []
    index = 0
    for cluster in used_clusters:
        point1 = cluster[name1]
        point2 = cluster[name2]
        kps1.append(cv2.KeyPoint(point1["x"], point1["y"], point1["size"]))
        kps2.append(cv2.KeyPoint(point2["x"], point2["y"], point2["size"]))
        matches.append(cv2.DMatch(index, index, 0))
        index += 1

    img3 = cv2.drawMatches(img1, kps1, img2, kps2,
                           matches, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()


def find_shift(project: vstarstack.tool.cfg.Project, argv: list):
    """Display clusters"""
    if len(argv) >= 2:
        clusters_f = argv[0]
        shifts_f = argv[1]
    else:
        clusters_f = project.config.cluster.path
        shifts_f = project.config.paths.relative_shifts
    with open(clusters_f, encoding='utf8') as f:
        clusters = json.load(f)
    shifts = build_movements(Movement, clusters)
    serialized = {}
    for name1,shifts1 in shifts.items():
        serialized[name1] = {}
        for name2 in shifts1:
            serialized[name1][name2] = shifts[name1][name2].serialize()
    with open(shifts_f, "w", encoding='utf8') as f:
        json.dump(serialized, f, ensure_ascii=False, indent=4)

commands = {
    "display": (display,
                "Display clusters",
                "cluster.json channel file1.zip file2.zip"),
    "find-shift": (find_shift,
                   "Find shifts from cluster file",
                   "cluster.json shifts.json"),
}

def run(project: vstarstack.tool.cfg.Project, argv: list):
    vstarstack.tool.usage.run(project, argv, "cluster", commands, autohelp=True)
