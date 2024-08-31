"""Remove hot pixels"""
#
# Copyright (c) 2024 Vladislav Tsendrovskii
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
from vstarstack.library.data import DataFrame

def remove_hot_pixels(image : DataFrame) -> DataFrame:
    """
    Remove hot pixels from image
    
    
    """
    for channel in image.get_channels():
        layer, opts = image.get_channel(channel)
        if not image.get_channel_option(channel, "signal"):
            continue
        hotpixels = np.where(layer >= np.median(layer)*3)
        layer[hotpixels] = 0
        image.add_channel(layer, channel, **opts)

        weight, _, weight_channel = image.get_linked_channel(channel, "weight")
        if weight_channel is not None:
            weight[hotpixels] = 0
            image.add_channel(weight, weight_channel, weight=True)
    return image
