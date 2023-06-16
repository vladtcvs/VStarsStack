"""Normalize image layers"""
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
import vstarstack.library.common
import vstarstack.library.data

def normalize(dataframe : vstarstack.library.data.DataFrame):
    """Normalize image layers"""
    for channel in dataframe.get_channels():
        image, opts = dataframe.get_channel(channel)
        if "normalized" in opts and opts["normalized"]:
            continue
        if opts["weight"]:
            continue
        if opts["encoded"]:
            continue
        if channel not in dataframe.links["weight"]:
            continue
        weight, _ = dataframe.get_channel(dataframe.links["weight"][channel])
        image = image / weight
        image[np.where(weight == 0)] = 0
        opts["normalized"] = True
        dataframe.add_channel(image, channel, **opts)

    return dataframe
