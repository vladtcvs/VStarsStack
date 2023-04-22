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
import cv2
import matplotlib.pyplot as plt
import numpy as np

import vstarstack.data
import vstarstack.cfg


def run(_project: vstarstack.cfg.Project, argv: list):
    with open(argv[0], encoding='utf8') as file:
        clusters = json.load(file)

    fname1 = argv[2]
    fname2 = argv[3]
    channel = argv[1]

    df1 = vstarstack.data.DataFrame.load(fname1)
    df2 = vstarstack.data.DataFrame.load(fname2)

    img1, _ = df1.get_channel(channel)
    img2, _ = df2.get_channel(channel)

    img1 = (img1 / np.amax(img1) * 255).astype(np.uint8)
    img2 = (img2 / np.amax(img2) * 255).astype(np.uint8)

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
    print(len(used_clusters))
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
