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
import vstarstack.library.data
import vstarstack.library.common
import vstarstack.library.image_process.normalize
from vstarstack.library.data import DataFrame
from copy import deepcopy

def simple_add(images : vstarstack.library.common.IImageSource) -> DataFrame:
    """Just add images"""

    summary = {}
    params = {}
    summary_weight = {}

    if images.empty():
        return None

    channel_opts = {}
    for img in images.items():
        params = img.params
        img = vstarstack.library.image_process.normalize.denormalize(img)
        for channel_name in img.get_channels():
            channel, opts = img.get_channel(channel_name)
            if not img.get_channel_option(channel_name, "brightness"):
                continue
            if channel_name not in channel_opts:
                channel_opts[channel_name] = opts

            weight, _, _ = img.get_linked_channel(channel_name, "weight")
            if weight is None:
                if (weight_k := img.get_parameter("weight")) is None:
                    weight_k = 1
                weight = np.ones(channel.shape, dtype=np.float64) * weight_k

            if channel_name not in summary:
                summary[channel_name] = deepcopy(channel.astype(np.float64))
                summary_weight[channel_name] = deepcopy(weight)
            else:
                try:
                    summary[channel_name] += channel
                    summary_weight[channel_name] += weight
                except Exception:
                    print("Can not add image. Skipping")

    result = vstarstack.library.data.DataFrame(params=params)
    for channel_name, channel in summary.items():
        print(channel_name)
        weight_channel_name = "weight-"+channel_name
        result.add_channel(channel, channel_name, **channel_opts[channel_name])
        result.add_channel(summary_weight[channel_name], weight_channel_name, weight=True)
        result.add_channel_link(channel_name, weight_channel_name, "weight")

    return result
