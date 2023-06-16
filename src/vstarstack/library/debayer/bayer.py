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
    mask = np.array([
        [[0, 0], [0, 0]],  # red
        [[0, 0], [0, 0]],  # green
        [[0, 0], [0, 0]],  # blue
    ])
    if name[0] == "R":
        mask[0][0][0] = 1
    elif name[0] == "G":
        mask[1][0][0] = 1
    elif name[0] == "B":
        mask[2][0][0] = 1

    if name[1] == "R":
        mask[0][0][1] = 1
    elif name[1] == "G":
        mask[1][0][1] = 1
    elif name[1] == "B":
        mask[2][0][1] = 1

    if name[2] == "R":
        mask[0][1][0] = 1
    elif name[2] == "G":
        mask[1][1][0] = 1
    elif name[2] == "B":
        mask[2][1][0] = 1

    if name[3] == "R":
        mask[0][1][1] = 1
    elif name[3] == "G":
        mask[1][1][1] = 1
    elif name[3] == "B":
        mask[2][1][1] = 1

    return mask

def _getcolor(img, mask):
    return np.sum(img*mask)

def debayer_image(image : np.ndarray,
                  weight : np.ndarray,
                  mask : np.ndarray):
    """Process debayer on image"""
    h = image.shape[0]
    w = image.shape[1]

    cshape = (int(h/2), int(w/2))

    R = np.zeros(cshape)
    G = np.zeros(cshape)
    B = np.zeros(cshape)

    w_R = np.zeros(cshape)
    w_G = np.zeros(cshape)
    w_B = np.zeros(cshape)

    for y in range(int(h/2)):
        for x in range(int(w/2)):
            cut = image[2*y:2*y+2, 2*x:2*x+2]
            wcut = weight[2*y:2*y+2, 2*x:2*x+2]
            R[y][x] = _getcolor(cut, mask[0])
            G[y][x] = _getcolor(cut, mask[1])
            B[y][x] = _getcolor(cut, mask[2])

            w_R[y][x] = _getcolor(wcut, mask[0])
            w_G[y][x] = _getcolor(wcut, mask[1])
            w_B[y][x] = _getcolor(wcut, mask[2])

    return R, G, B, w_R, w_G, w_B

def debayer_dataframe(dataframe : vstarstack.library.data.DataFrame,
                      mask : np.ndarray,
                      raw_channel_name : str):
    """Debayer dataframe"""
    raw, _ = dataframe.get_channel(raw_channel_name)
    weight, _ = dataframe.get_channel(dataframe.links["weight"][raw_channel_name])

    R, G, B, w_R, w_G, w_B = debayer_image(raw, weight, mask)
    dataframe.add_channel(R, "R", brightness=True)
    dataframe.add_channel(G, "G", brightness=True)
    dataframe.add_channel(B, "B", brightness=True)

    dataframe.add_channel(w_R, "weight-R", weight=True)
    dataframe.add_channel(w_G, "weight-G", weight=True)
    dataframe.add_channel(w_B, "weight-B", weight=True)

    dataframe.add_channel_link("R", "weight-R", "weight")
    dataframe.add_channel_link("G", "weight-G", "weight")
    dataframe.add_channel_link("B", "weight-B", "weight")

    return dataframe
