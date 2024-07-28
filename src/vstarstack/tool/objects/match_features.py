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

import math
import json
import numpy as np

import vstarstack.tool.common
import vstarstack.tool.cfg
import vstarstack.library.data
from vstarstack.library.objects.features import build_clusters

def load_keypoints(files):
    points = {}
    for file in files:
        with open(file) as f:
            record = json.load(f)
            name = record["name"]
            points[name] = record
    return points

def run(project: vstarstack.tool.cfg.Project, argv: list[str]):
    if len(argv) >= 2:
        points_path = argv[0]
        clusters_fname = argv[1]
    else:
        points_path = project.config.paths.npy_fixed
        clusters_fname = project.config.cluster.path

    max_feature_delta = project.config.objects.features.max_feature_delta
    features_percent = project.config.objects.features.features_percent / 100.0

    files = vstarstack.tool.common.listfiles(points_path, ".json")
    files = [filename for _, filename in files]
    keypoints = load_keypoints(files)
    print("Found keypoints")

    points = {}
    descs = {}

    for name in keypoints:
        points[name] = [item["keypoint"] for item in keypoints[name]["points"]]
        descs[name] = np.array([np.array(item["descriptor"], dtype=np.uint8) for item in keypoints[name]["points"]])

    clusters = build_clusters(points, descs,
                              max_feature_delta,
                              features_percent)

    print("Builded clusters")
    vstarstack.tool.common.check_dir_exists(clusters_fname)
    with open(clusters_fname, "w") as f:
        json.dump(clusters, f, indent=4, ensure_ascii=False)
