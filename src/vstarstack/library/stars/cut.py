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
import cv2

import vstarstack.library.data

def cut_stars(image : vstarstack.library.data.DataFrame,
              descs : list):
    """Remove stars from image"""
    new_image = vstarstack.library.data.DataFrame(params=image.params)
    for name in image.get_channels():
        layer, opts = image.get_channel(name)
        if not opts["brightness"]:
            continue
        mask = np.zeros(layer.shape)
        for star in descs:
            x = int(star["x"]+0.5)
            y = int(star["y"]+0.5)
            r = int(star["radius"]+0.5)
            cv2.circle(mask, (x,y), r*2+10, 1, -1)
        mask = 1-mask

        layer = layer * mask
        new_image.add_channel(layer, name, brightness=True)
        if name in image.links["weight"]:
            weight,_ = image.get_channel(image.links["weight"][name])
            weight = weight * mask
            new_image.add_channel(weight, image.links["weight"][name], weight=True)
            new_image.add_channel_link(name, image.links["weight"][name], "weight")
    return new_image
