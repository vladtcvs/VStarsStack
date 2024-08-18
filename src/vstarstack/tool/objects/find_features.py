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
import cv2

import vstarstack.tool.common
import vstarstack.tool.cfg
import vstarstack.library.data
import vstarstack.library.common
from vstarstack.library.objects.features import find_keypoints_orb
from vstarstack.library.objects.features import find_keypoints_brightness
from vstarstack.library.objects.features import describe_keypoints

def _save_features(points, name, features_path):
    fname = os.path.join(features_path, f"{name}_keypoints.json")
    vstarstack.tool.common.check_dir_exists(fname)
    with open(fname, "w") as f:
        json.dump(points, f, indent=4, ensure_ascii=False)

def build_keypoints_structure(keypoints, ds, fname, name):
    record = {
        "fname" : fname,
        "name" : name,
        "points" : [],
    }

    for keypoint, desc in zip(keypoints, ds):
        for di in desc:
            if di != 0:
                break
        else:
            continue
        record["points"].append({
            "keypoint" : keypoint,
            "descriptor" : [int(item) for item in list(desc)],
        })

    return record

def _proj_find_keypoints_orb(files, num_splits, param, features_path):
    for name, fname in files:
        print(f"Processing {name}")
        dataframe = vstarstack.library.data.DataFrame.load(fname)
        gray, _ = vstarstack.library.common.df_to_light(dataframe)

        keypoints = find_keypoints_orb(gray, num_splits, param)
        ds = describe_keypoints(gray, keypoints, param)
        points = build_keypoints_structure(keypoints, ds, fname, name)

        _save_features(points, name, features_path)

def _proj_find_keypoints_brightness(files, num_splits, param, orb_param, features_path):
    for name, fname in files:
        print(f"Processing {name}")
        dataframe = vstarstack.library.data.DataFrame.load(fname)
        gray, _ = vstarstack.library.common.df_to_light(dataframe)

        keypoints = find_keypoints_brightness(gray, num_splits, param)
        ds = describe_keypoints(gray, keypoints, orb_param)
        points = build_keypoints_structure(keypoints, ds, fname, name)

        _save_features(points, name, features_path)


def find_points_orb(project: vstarstack.tool.cfg.Project, argv: list[str]):
    if len(argv) >= 2:
        inputs = argv[0]
        features = argv[1]
    else:
        inputs = project.config.paths.npy_fixed
        features = project.config.objects.features.path

    num_splits = project.config.objects.features.num_splits
    param = {
        "patchSize" : project.config.objects.features.orb.patchSize
    }

    files = vstarstack.tool.common.listfiles(inputs, ".zip")
    _proj_find_keypoints_orb(files, num_splits, param, features)

def find_points_brightness(project: vstarstack.tool.cfg.Project, argv: list[str]):
    if len(argv) >= 2:
        inputs = argv[0]
        features = argv[1]
    else:
        inputs = project.config.paths.npy_fixed
        features = project.config.objects.features.path

    num_splits = project.config.objects.features.num_splits

    orb_param = {
        "patchSize" : project.config.objects.features.orb.patchSize
    }

    param = {
            "blur_size" : project.config.objects.features.bright_spots.blurSize,
            "k_thr" : project.config.objects.features.bright_spots.k_thr,
            "min_value" : project.config.objects.features.bright_spots.minValue,
            "min_pixel" : project.config.objects.features.bright_spots.minPixel,
            "max_pixel" : project.config.objects.features.bright_spots.maxPixel,
        }

    files = vstarstack.tool.common.listfiles(inputs, ".zip")
    _proj_find_keypoints_brightness(files, num_splits, param, orb_param, features)

def display_features(project: vstarstack.tool.cfg.Project, argv: list[str]):
    image_fname = argv[0]
    features_fname = argv[1]
    print(f"{image_fname} - {features_fname}")
    df = vstarstack.library.data.DataFrame.load(image_fname)
    light, _ = vstarstack.library.common.df_to_light(df)
    keypoints = []
    
    cv2.drawKeypoints(light, keypoints)

commands = {
    "brightness": (find_points_brightness, "find keypoints with brightness detector", "npys/ features/"),
    "orb": (find_points_orb, "find keypoints with ORB detector", "npys/ features/")
}

