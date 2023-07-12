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

def border(dataframe : vstarstack.library.data.DataFrame,
           bw_left : int, bw_top : int,
           bw_right : int, bw_bottom : int):
    """Remove image borders"""

    for channel in dataframe.get_channels():
        image, opts = dataframe.get_channel(channel)
        if opts["encoded"]:
            continue
        if opts["weight"]:
            continue

        w_channel = dataframe.links["weight"][channel]
        weight, _ = dataframe.get_channel(w_channel)

        w = image.shape[1]
        h = image.shape[0]

        image[0:bw_top, :] = 0
        image[(h-bw_bottom):h, :] = 0

        image[:, 0:bw_left] = 0
        image[:, (w-bw_right):w] = 0

        weight[0:bw_top, :] = 0
        weight[(h-bw_bottom):h, :] = 0

        weight[:, 0:bw_left] = 0
        weight[:, (w-bw_right):w] = 0

        dataframe.add_channel(image, channel, **opts)
        dataframe.add_channel(weight, w_channel, weight=True)
        dataframe.add_channel_link(channel, w_channel, "weight")

    return dataframe
