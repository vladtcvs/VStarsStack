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

import vstarstack.tool.common
import vstarstack.tool.cfg
import vstarstack.library.data
from vstarstack.library.objects.features import build_keypoints
from vstarstack.library.objects.features import build_clusters


def find_keypoints(files, num_splits, detector_type, param):
    points = {}
    descs = {}
    fnames = {}

    for fname in files:
        name = os.path.splitext(os.path.basename(fname))[0]
        #print(name)
        fnames[name] = fname
        dataframe = vstarstack.library.data.DataFrame.load(fname)
        for channel in dataframe.get_channels():
            image, opts = dataframe.get_channel(channel)
            if not opts["brightness"]:
                continue
            if channel not in points:
                points[channel] = {}
                descs[channel] = {}

            keypoints, keydescs = build_keypoints(image,
                                        num_splits,
                                        detector_type,
                                        param)

            points[channel][name] = keypoints
            descs[channel][name] = keydescs

    return points, descs, fnames

def run(project: vstarstack.tool.cfg.Project, argv: list[str]):
    if len(argv) >= 2:
        inputs = argv[0]
        clusters_fname = argv[1]
    else:
        inputs = project.config.paths.npy_fixed
        clusters_fname = project.config.cluster.path

    detector_type = project.config.objects.features.detector
    num_splits = project.config.objects.features.num_splits
    max_feature_delta = project.config.objects.features.max_feature_delta
    features_percent = project.config.objects.features.features_percent / 100

    if detector_type == "orb":
        param = None
        print("Using ORB detector")
    elif detector_type == "brightness":
        param = {
            "blur_size" : project.config.objects.features.bright_spots.blurSize,
            "k_thr" : project.config.objects.features.bright_spots.k_thr,
            "min_value" : project.config.objects.features.bright_spots.minValue,
            "min_pixel" : project.config.objects.features.bright_spots.minPixel,
            "max_pixel" : project.config.objects.features.bright_spots.maxPixel,
        }
        print("Using brightness detector")

    files = vstarstack.tool.common.listfiles(inputs, ".zip")
    files = [filename for _, filename in files]
    points, descs, _ = find_keypoints(files, num_splits, detector_type, param)
    print("Found keypoints")

    total_clusters = []
    for channel in points:
        print(f"\tBuild clusters for {channel} channel")
        crd_clusters = build_clusters(points[channel],
                                      descs[channel],
                                      max_feature_delta,
                                      features_percent)

        total_clusters += crd_clusters

    print("Builded clusters")
    vstarstack.tool.common.check_dir_exists(clusters_fname)
    with open(clusters_fname, "w") as f:
        json.dump(total_clusters, f, indent=4, ensure_ascii=False)
