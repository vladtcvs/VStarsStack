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

def normalize(dataframe : vstarstack.library.data.DataFrame, deepcopy=True):
    """Normalize image layers"""
    if deepcopy:
        new_dataframe = dataframe.copy()
    else:
        new_dataframe = dataframe
    for channel in new_dataframe.get_channels():
        image, _ = new_dataframe.get_channel(channel)
        if new_dataframe.get_channel_option(channel, "normed"):
            continue
        if not new_dataframe.get_channel_option(channel, "brightness"):
            continue
        weight, _, _ = new_dataframe.get_linked_channel(channel, "weight")
        if weight is None:
            continue

        image = image / weight
        image[np.where(weight < 1e-12)] = 0
        new_dataframe.replace_channel(image, channel, normed=True)

    return new_dataframe

def denormalize(dataframe : vstarstack.library.data.DataFrame, deepcopy=True):
    """De-normalize image layers"""
    if deepcopy:
        new_dataframe = dataframe.copy()
    else:
        new_dataframe = dataframe
    for channel in new_dataframe.get_channels():
        image, _ = new_dataframe.get_channel(channel)
        if not new_dataframe.get_channel_option(channel, "normed"):
            continue
        if not new_dataframe.get_channel_option(channel, "brightness"):
            continue
        weight, _, _ = new_dataframe.get_linked_channel(channel, "weight")
        if weight is None:
            continue

        image = image * weight
        new_dataframe.replace_channel(image, channel, normed=False)

    return new_dataframe
