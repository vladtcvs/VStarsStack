#
# Copyright (c) 2023-2024 Vladislav Tsendrovskii
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
import csv

import vstarstack.tool.cfg
import vstarstack.tool.usage

import vstarstack.library.data
import vstarstack.library.clusters.clusters
from vstarstack.library.movement.find_shift import build_movements, complete_movements
from vstarstack.library.movement.sphere import Movement

def display(_project: vstarstack.tool.cfg.Project, argv: list):
    """Display clusters"""
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

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
    compose = project.config.cluster.compose_movements
    if len(argv) >= 2:
        clusters_f = argv[0]
        shifts_f = argv[1]
        error_f = "shift_errors.csv"
    else:
        clusters_f = project.config.cluster.path
        shifts_f = project.config.paths.relative_shifts
        error_f = project.config.paths.shift_errors
    with open(clusters_f, encoding='utf8') as f:
        clusters = json.load(f)
    shifts, errors = build_movements(Movement, clusters, None)
    shifts = complete_movements(Movement, shifts, compose)
    serialized = {}
    for name1,shifts1 in shifts.items():
        serialized[name1] = {}
        for name2 in shifts1:
            serialized[name1][name2] = shifts[name1][name2].serialize()
    if len(errors) > 0:
        print("Couldn't build movement for pairs:")
        with open(error_f, "w", encoding='utf8') as f:
            writer = csv.writer(f)
            writer.writerow(["name2", "name1"])
            for name1, name2 in errors:
                print(f"\t{name2} -> {name1}")
                writer.writerow([name2,name1])
        
    with open(shifts_f, "w", encoding='utf8') as f:
        json.dump(serialized, f, ensure_ascii=False, indent=4)
    
def find_shift_to_selected(project: vstarstack.tool.cfg.Project, argv: list):
    """Display clusters"""
    compose = project.config.cluster.compose_movements
    if len(argv) >= 3:
        clusters_f = argv[0]
        shifts_f = argv[1]
        basic_image = argv[2]
        error_f = "shift_errors.csv"
    elif len(argv) >= 1:
        clusters_f = project.config.cluster.path
        shifts_f = project.config.paths.absolute_shifts
        error_f = project.config.paths.shift_errors
        basic_image = argv[0]
    else:
        print("Invalid args")
        return
    with open(clusters_f, encoding='utf8') as f:
        clusters = json.load(f)
    shifts, errors = build_movements(Movement, clusters, basic_image)
    shifts = complete_movements(Movement, shifts, compose)
    serialized = {}
    shifts1 = shifts[basic_image]
    for name2 in shifts1:
        serialized[name2] = shifts[basic_image][name2].serialize()

    if len(errors) > 0:
        print("Couldn't build movement for pairs:")
        with open(error_f, "w", encoding='utf8') as f:
            writer = csv.writer(f)
            writer.writerow(["name2", "name1"])
            for name1, name2 in errors:
                print(f"\t{name2} -> {name1}")
                writer.writerow([name2,name1])

    with open(shifts_f, "w", encoding='utf8') as f:
        json.dump(serialized, f, ensure_ascii=False, indent=4)

def _prepare_match_table(match_table):
    mt = {}
    for image_id1 in match_table:
        mt[image_id1] = {}
        for image_id2 in match_table[image_id1]:
            mt[image_id1][image_id2] = {}
            for star_id1 in match_table[image_id1][image_id2]:
                star_id2 = match_table[image_id1][image_id2][star_id1]
                mt[image_id1][image_id2][int(star_id1)] = star_id2
    return mt

def build_from_match_table(project: vstarstack.tool.cfg.Project, argv: list):
    """Build clusters file from match table"""
    if len(argv) >= 2:
        descs_path = argv[0]
        match_table_f = argv[1]
        cluster_f = argv[2]
    else:
        descs_path = project.config.paths.descs
        match_table_f = project.config.cluster.matchtable
        cluster_f = project.config.cluster.path

    with open(match_table_f, encoding='utf8') as f:
        match_table = _prepare_match_table(json.load(f))

    print("Find index cluster")
    clusters = vstarstack.library.clusters.clusters.find_clusters_in_match_table(match_table)
    dclusters = sorted(clusters, key=lambda x : len(x), reverse=True)
    dclusters = [item for item in dclusters if len(item) > 1]
    print("Done")

    stars_files = vstarstack.tool.common.listfiles(descs_path, ".json")
    descs = {}
    for name, fname in stars_files:
        with open(fname, encoding='utf8') as file:
            desc = json.load(file)
        descs[name] = desc

    star_clusters = []
    for cluster in dclusters:
        star_cluster = {}
        for name, star_id in cluster.items():
            star_cluster[name] = descs[name]["points"][star_id]["keypoint"]
        star_clusters.append(star_cluster)

    with open(cluster_f, "w", encoding='utf8') as f:
        json.dump(star_clusters, f, indent=4)

commands = {
    "display": (display,
                "Display clusters",
                "cluster.json channel file1.zip file2.zip"),
    "build-from-matchtable" : (build_from_match_table,
                               "Build clusters file from match table",
                               "descs/ match_table.json clusters.json"),
    "find-shifts": (find_shift,
                    "Find shifts from cluster file",
                    "cluster.json shifts.json"),
    "find-shift-to-selected": (find_shift_to_selected,
                   "Find shifts from cluster file, but only to selected image",
                   "clusters.json shift.json <basic_image>"),
}
