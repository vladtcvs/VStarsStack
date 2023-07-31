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
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2

import vstarstack.library.data
import vstarstack.tool.cfg
import vstarstack.tool.cfg

def show(project: vstarstack.tool.cfg.Project, argv: list):
    """Display detected stars"""
    image = vstarstack.library.data.DataFrame.load(argv[0])
    channel = argv[1]
    with open(argv[2], encoding='utf8') as f:
        descs = json.load(f)
    slope = vstarstack.tool.cfg.get_param("multiply", float, 1)
    layer, _ = image.get_channel(channel)
    layer = np.clip(layer/np.amax(layer)*slope, 0, 1)
    layer = (layer * 255).astype(np.uint8)

    showed = np.zeros((layer.shape[0], layer.shape[1], 3), dtype="uint8")
    showed[:,:,0] = layer
    showed[:,:,1] = layer
    showed[:,:,2] = layer
    for star in descs["stars"]:
        cv2.circle(showed, (star["x"], star["y"]), int(star["size"]+11), (255,0,0), 1)
    plt.imshow(showed)
    plt.show()

def get_brightness(df):
    light = None
    for channel in df.get_channels():
        layer, opts = df.get_channel(channel)
        if not opts["brightness"]:
            continue
        if light is None:
            light = layer
        else:
            light = light + layer
    return light

def show_match(project: vstarstack.tool.cfg.Project, argv: list):
    """Display detected stars"""
    fname1 = argv[0]
    fname2 = argv[1]
    name1 = os.path.splitext(os.path.basename(fname1))[0]
    name2 = os.path.splitext(os.path.basename(fname2))[0]
    image1 = vstarstack.library.data.DataFrame.load(fname1)
    image2 = vstarstack.library.data.DataFrame.load(fname2)
    with open(argv[2], encoding='utf8') as f:
        descs1 = json.load(f)
    with open(argv[3], encoding='utf8') as f:
        descs2 = json.load(f)
    with open(argv[4], encoding='utf8') as f:
        match_table = json.load(f)

    slope = vstarstack.tool.cfg.get_param("multiply", float, 1)
    layer1 = get_brightness(image1)
    layer2 = get_brightness(image2)
    layer1 = np.clip(layer1/np.amax(layer1)*slope, 0, 1)
    layer1 = (layer1 * 255).astype(np.uint8)
    layer2 = np.clip(layer2/np.amax(layer2)*slope, 0, 1)
    layer2 = (layer2 * 255).astype(np.uint8)

    kps1 = []
    kps2 = []
    matches = []
    index = 0
    for star1_id, star2_id in match_table[name1][name2].items():
        star1_id = int(star1_id)
        star1 = descs1["main"][star1_id]["star"]
        star2 = descs2["main"][star2_id]["star"]
        kps1.append(cv2.KeyPoint(star1["x"], star1["y"], int(star1["radius"])+2))
        kps2.append(cv2.KeyPoint(star2["x"], star2["y"], int(star2["radius"])+2))
        matches.append(cv2.DMatch(index, index, 0))
        index += 1

    img3 = cv2.drawMatches(layer1, kps1, layer2, kps2,
                           matches, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()
