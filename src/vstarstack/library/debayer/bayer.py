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

import numpy as np
import vstarstack.library.data

def generate_mask(name):
    """Generate mask by name"""
    used_colors = set(name)
    mask = {}
    for color in used_colors:
        mask[color] = [[0, 0], [0, 0]]

    pixels = ((0,0),(0,1),(1,0),(1,1))

    for i, crd in enumerate(pixels):
        mask[name[i]][crd[0]][crd[1]] = 1

    result_mask = {}
    for color in used_colors:
        color_mask = np.array(mask[color])
        color_mask = color_mask / np.sum(color_mask)
        result_mask[color] = color_mask
    return result_mask

def debayer_image(image : np.ndarray,
                  weight : np.ndarray,
                  mask : dict):
    """Process debayer on image"""
    h = image.shape[0]
    w = image.shape[1]

    cshape = (int(h/2), int(w/2))

    layers = {}
    weights = {}
    for color in mask:
        layers[color] = np.zeros(cshape)
        weights[color] = np.zeros(cshape)

    if h % 2 == 1:
        image = image[0:h-1,:]
        weight = weight[0:h-1,:]
        h -= 1
    if w % 2 == 1:
        image = image[:,0:w-1]
        weight = weight[:,0:w-1]
        w -= 1

    for color in mask:
        k00 = mask[color][0,0]
        k01 = mask[color][0,1]
        k10 = mask[color][1,0]
        k11 = mask[color][1,1]
        image00 = image[0::2, 0::2]
        image01 = image[0::2, 1::2]
        image10 = image[1::2, 0::2]
        image11 = image[1::2, 1::2]
        weight00 = weight[0::2, 0::2]
        weight01 = weight[0::2, 1::2]
        weight10 = weight[1::2, 0::2]
        weight11 = weight[1::2, 1::2]

        layers[color] = k00 * image00 + k01 * image01 + k10 * image10 + k11 * image11
        weights[color] = k00 * weight00 + k01 * weight01 + k10 * weight10 + k11 * weight11

    return layers, weights

def debayer_dataframe(dataframe : vstarstack.library.data.DataFrame,
                      mask : dict,
                      raw_channel_name : str):
    """Debayer dataframe"""
    raw, _ = dataframe.get_channel(raw_channel_name)
    weight, _ = dataframe.get_channel(dataframe.links["weight"][raw_channel_name])

    layers, weights = debayer_image(raw, weight, mask)
    for color in layers:
        dataframe.add_channel(layers[color], color, brightness=True)
        dataframe.add_channel(weights[color], f"weight-{color}", weight=True)
        dataframe.add_channel_link(color, f"weight-{color}", "weight")

    dataframe.remove_channel(dataframe.links["weight"][raw_channel_name])
    dataframe.remove_channel(raw_channel_name)

    dataframe.params["w"] = int(dataframe.params["w"]/2)
    dataframe.params["h"] = int(dataframe.params["h"]/2)
    dataframe.params["format"] = "flat"
    if "projection" in dataframe.params and dataframe.params["projection"] == "perspective":
        dataframe.params["perspective_kw"] *= 2
        dataframe.params["perspective_kh"] *= 2
    return dataframe
