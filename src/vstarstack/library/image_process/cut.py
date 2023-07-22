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
    w = right - left
    h = bottom - top
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

    result.params["w"] = w
    result.params["h"] = h
    return result
