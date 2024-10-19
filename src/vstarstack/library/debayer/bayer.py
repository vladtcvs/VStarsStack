#
# Copyright (c) 2022-2024 Vladislav Tsendrovskii
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
from enum import Enum

import vstarstack.library.data
from vstarstack.library.common import getpixel_none

class DebayerMethod(Enum):
    """
    Debayer methods:

        SUBSAMPLE - convert each 2*2 pixels square to 1 colour pixel
        MASK - separate image to color layers according to mask, and layer has weight=0 in points of other colors
        INTERPOLATE - interpolate missing colors of each pixel from neighbours
    """
    SUBSAMPLE = 0,
    CFA = 1,
    INTERPOLATE = 2,

def generate_mask(name : str):
    """Generate mask by name"""
    items = name.split("_")
    w = int(items[0])
    h = int(items[1])

    if w != 2 or h != 2:
        raise NotImplementedError("Now only 2x2 bayer filters are supported")

    colors = items[2]
    used_colors = set(colors)
    mask = {}
    for color in used_colors:
        mask[color] = np.zeros((h,w))

    pixels = []
    for y in range(h):
        for x in range(w):
            pixels.append((y,x))

    for i, crd in enumerate(pixels):
        mask[colors[i]][crd[0]][crd[1]] = 1

    result_mask = {}
    for color in used_colors:
        color_mask = np.array(mask[color])
        result_mask[color] = color_mask
    return result_mask

def debayer_image_subsample(image : np.ndarray,
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

def debayer_image_mask(image : np.ndarray,
                       weight : np.ndarray,
                       mask : dict):
    """Process debayer on image"""

    image00 = np.copy(image)
    weight00 = np.copy(weight)
    image00[0::2, 1::2] = 0
    weight00[0::2, 1::2] = 0
    image00[1::2, 0::2] = 0
    weight00[1::2, 0::2] = 0
    image00[1::2, 1::2] = 0
    weight00[1::2, 1::2] = 0

    image01 = np.copy(image)
    weight01 = np.copy(weight)
    image01[0::2, 0::2] = 0
    weight01[0::2, 0::2] = 0
    image01[1::2, 0::2] = 0
    weight01[1::2, 0::2] = 0
    image01[1::2, 1::2] = 0
    weight01[1::2, 1::2] = 0

    image10 = np.copy(image)
    weight10 = np.copy(weight)
    image10[0::2, 0::2] = 0
    weight10[0::2, 0::2] = 0
    image10[0::2, 1::2] = 0
    weight10[0::2, 1::2] = 0
    image10[1::2, 1::2] = 0
    weight10[1::2, 1::2] = 0

    image11 = np.copy(image)
    weight11 = np.copy(weight)
    image11[0::2, 0::2] = 0
    weight11[0::2, 0::2] = 0
    image11[0::2, 1::2] = 0
    weight11[0::2, 1::2] = 0
    image11[1::2, 0::2] = 0
    weight11[1::2, 0::2] = 0

    layers = {}
    weights = {}

    for color in mask:
        k00 = mask[color][0,0]
        k01 = mask[color][0,1]
        k10 = mask[color][1,0]
        k11 = mask[color][1,1]

        layers[color] = k00 * image00 + k01 * image01 + k10 * image10 + k11 * image11
        weights[color] = k00 * weight00 + k01 * weight01 + k10 * weight10 + k11 * weight11

    return layers, weights

def debayer_image_interpolate(image : np.ndarray,
                              weight : np.ndarray,
                              mask : dict):
    h = image.shape[0]
    w = image.shape[1]
    layers, weights = debayer_image_mask(image, weight, mask)
    for color in layers:
        layer = layers[color]
        weight = weights[color]
        
        newlayer = np.copy(layer)
        newweight = np.copy(weight)

        idx = np.where(weight < 1e-12)
        newlayer[idx] = 0
        newweight[idx] = 0
        count_used = np.zeros(layer.shape)

        shifts = [(-1,-1), (-1,0), (-1,1),
                  (0,-1), (0,1),
                  (1,-1),(1,0),(1,1)]

        shifted = []
        for dy,dx in shifts:
            sh_layer = np.roll(layer, (dy, dx), (0, 1))
            sh_weight = np.roll(weight, (dy, dx), (0,1))
            shifted.append((sh_layer, sh_weight))

        for sh_layer, sh_weight in shifted:
            use_pixels = (sh_weight > 1e-12).astype('uint8')
            newlayer[idx] = newlayer[idx] + sh_layer[idx] * sh_weight[idx] * use_pixels[idx]
            newweight[idx] = newweight[idx] + sh_weight[idx] * use_pixels[idx]
            count_used[idx] += use_pixels[idx]

        newlayer[idx] = newlayer[idx] / newweight[idx]
        newweight[idx] = newweight[idx] / count_used[idx]
        newweight = np.nan_to_num(newweight, nan=0, posinf=0, neginf=0)
        newlayer[np.where(newweight == 0)] = 0

        layers[color] = newlayer
        weights[color] = newweight
    return layers, weights

def debayer_dataframe(dataframe : vstarstack.library.data.DataFrame,
                      mask : dict,
                      raw_channel_name : str,
                      method : DebayerMethod):
    """Debayer dataframe"""
    raw, _ = dataframe.get_channel(raw_channel_name)
    weight, _, _ = dataframe.get_linked_channel(raw_channel_name, "weight")
    normed = dataframe.get_channel_option(raw_channel_name, "normed")
    if weight is None:
        if (weight_val := dataframe.get_parameter("weight")) is None:
            weight_val = 1
        weight = np.ones(raw.shape)*weight_val

    if normed:
        raw = raw * weight

    if method == DebayerMethod.SUBSAMPLE:
        layers, weights = debayer_image_subsample(raw, weight, mask)
    elif method == DebayerMethod.CFA:
        layers, weights = debayer_image_mask(raw, weight, mask)
    elif method == DebayerMethod.INTERPOLATE:
        layers, weights = debayer_image_interpolate(raw, weight, mask)
    else:
        return None

    for color in layers:
        dataframe.add_channel(layers[color], color, brightness=True, signal=True, normed=False, cfa=True)
        dataframe.add_channel(weights[color], f"weight-{color}", weight=True)
        dataframe.add_channel_link(color, f"weight-{color}", "weight")

    if raw_channel_name in dataframe.links["weight"]:
        dataframe.remove_channel(dataframe.links["weight"][raw_channel_name])
    dataframe.remove_channel(raw_channel_name)

    if method == DebayerMethod.SUBSAMPLE:
        dataframe.add_parameter(int(dataframe.get_parameter("w")/2), "w")
        dataframe.add_parameter(int(dataframe.get_parameter("h")/2), "h")
        dataframe.add_parameter("flat", "format")
        if dataframe.get_parameter("projection") == "perspective":
            dataframe.params["projection_perspective_kw"] *= 2
            dataframe.params["projection_perspective_kh"] *= 2

    return dataframe
