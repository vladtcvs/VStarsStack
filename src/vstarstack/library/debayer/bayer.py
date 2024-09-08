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
                  weight : float,
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
        h -= 1
    if w % 2 == 1:
        image = image[:,0:w-1]
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
        weight00 = np.ones(image00.shape)*weight
        weight01 = np.ones(image01.shape)*weight
        weight10 = np.ones(image10.shape)*weight
        weight11 = np.ones(image11.shape)*weight

        layers[color] = k00 * image00 + k01 * image01 + k10 * image10 + k11 * image11
        weights[color] = k00 * weight00 + k01 * weight01 + k10 * weight10 + k11 * weight11

    return layers, weights

def debayer_dataframe(dataframe : vstarstack.library.data.DataFrame,
                      mask : dict,
                      raw_channel_name : str):
    """Debayer dataframe"""
    raw, _ = dataframe.get_channel(raw_channel_name)
    if (weight := dataframe.get_parameter("weight")) is None:
        weight = 1

    layers, weights = debayer_image(raw, weight, mask)
    for color in layers:
        dataframe.add_channel(layers[color], color, brightness=True, signal=True)
        dataframe.add_channel(weights[color], f"weight-{color}", weight=True)
        dataframe.add_channel_link(color, f"weight-{color}", "weight")

    if raw_channel_name in dataframe.links["weight"]:
        dataframe.remove_channel(dataframe.links["weight"][raw_channel_name])
    dataframe.remove_channel(raw_channel_name)

    dataframe.add_parameter(int(dataframe.get_parameter("w")/2), "w")
    dataframe.add_parameter(int(dataframe.get_parameter("h")/2), "h")
    dataframe.add_parameter("flat", "format")
    if dataframe.get_parameter("projection") == "perspective":
        dataframe.params["projection_perspective_kw"] *= 2
        dataframe.params["projection_perspective_kh"] *= 2
    return dataframe
