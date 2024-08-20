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

import csv
import json
import numpy as np

import vstarstack.tool.common
import vstarstack.tool.cfg
import vstarstack.library.data
from vstarstack.library.objects.features import match_images

def _load_keypoints(files):
    points = {}
    for file in files:
        with open(file, encoding='utf8') as f:
            record = json.load(f)
            name = record["name"]
            points[name] = record
    return points

def _build_default_match_table(names : list[str]):
    matches = []
    for name1 in names:
        for name2 in names:
            if name1 >= name2:
                continue
            matches.append((name1, name2))
    return matches

def _load_match_table(fname : str):
    match_table = []
    with open(fname, encoding='utf8') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            im1 = line[0]
            im2 = line[1]
            if im1 < im2:
                match_table.append((im1, im2))
            else:
                match_table.append((im2, im1))
    return match_table

def run(project: vstarstack.tool.cfg.Project, argv: list[str]):
    comparsion_list_file = None
    if len(argv) >= 2:
        points_path = argv[0]
        matchtable_fname = argv[1]
        if len(argv) >= 3:
            comparsion_list_file = argv[2]
    else:
        points_path = project.config.objects.features.path
        matchtable_fname = project.config.cluster.path

    max_feature_delta = project.config.objects.features.max_feature_delta
    features_percent = project.config.objects.features.features_percent / 100.0

    files = vstarstack.tool.common.listfiles(points_path, ".json")
    print(f"Found {len(files)} files")
    files = [filename for _, filename in files]
    keypoints = _load_keypoints(files)
    print("Found keypoints")

    points = {}
    descs = {}

    for name, record_points in keypoints.items():
        points[name] = [item["keypoint"] for item in record_points["points"]]
        descs[name] = np.array([np.array(item["descriptor"], dtype=np.uint8) for item in record_points["points"]])

    if comparsion_list_file is not None:
        match_list = _load_match_table(comparsion_list_file)
        print(f"Load comparsion list from {comparsion_list_file}: {len(match_list)} comparsions")
    else:
        match_list = _build_default_match_table(keypoints.keys())
        print(f"Build default comparsion list: {len(match_list)} comparsions")

    matches = match_images(points, descs,
                           max_feature_delta,
                           features_percent,
                           match_list)

    print("Builded match table")
    vstarstack.tool.common.check_dir_exists(matchtable_fname)
    with open(matchtable_fname, "w", encoding='utf8') as f:
        json.dump(matches, f, indent=4, ensure_ascii=False)
