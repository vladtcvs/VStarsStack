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

import vstarstack.library.data

def cut(dataframe : vstarstack.library.data.DataFrame,
           left : int, top : int,
           right : int, bottom : int):
    """Cut part of image"""

    result = vstarstack.library.data.DataFrame(params=dataframe.params)
    original_cox = 0
    original_coy = 0
    if "center_offset_x" in dataframe.params:
        original_cox = dataframe.params["center_offset_x"]
    if "center_offset_y" in dataframe.params:
        original_coy = dataframe.params["center_offset_y"]

    original_w = dataframe.params["w"]
    original_h = dataframe.params["h"]
    cutted_w = right - left
    cutted_h = bottom - top
    assert cutted_w > 0
    assert cutted_h > 0
    for channel in dataframe.get_channels():
        image, opts = dataframe.get_channel(channel)
        if opts["encoded"]:
            continue
        if opts["weight"]:
            continue

        w_channel = dataframe.links["weight"][channel]
        weight, _ = dataframe.get_channel(w_channel)

        image = image[top:bottom,left:right]
        weight = weight[top:bottom,left:right]

        result.add_channel(image, channel, **opts)
        result.add_channel(weight, w_channel, weight=True)
        result.add_channel_link(channel, w_channel, "weight")

    result.params["w"] = cutted_w
    result.params["h"] = cutted_h
    result.params["center_offset_x"] = (left + right)/2 - original_w / 2 + original_cox
    result.params["center_offset_y"] = (top + bottom)/2 - original_h / 2 + original_coy
    return result
