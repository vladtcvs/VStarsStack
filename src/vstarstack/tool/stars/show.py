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

import numpy as np
import json
import matplotlib.pyplot as plt
import cv2

import vstarstack.library.data
import vstarstack.tool.cfg
import vstarstack.tool.cfg

def run(project: vstarstack.tool.cfg.Project, argv: list):
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
