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

def _getcolor(img, mask):
    return np.sum(img*mask)

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

    for y in range(int(h/2)):
        for x in range(int(w/2)):
            cut = image[2*y:2*y+2, 2*x:2*x+2]
            wcut = weight[2*y:2*y+2, 2*x:2*x+2]
            for color in mask:
                layers[color][y][x] = _getcolor(cut, mask[color])
                weights[color][y][x] = _getcolor(wcut, mask[color])

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

    return dataframe
