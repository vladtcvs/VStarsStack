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
from vstarstack.library.data import DataFrame
from copy import deepcopy

def simple_add(images : vstarstack.library.common.IImageSource) -> DataFrame:
    """Just add images"""

    summary = {}
    summary_weight = {}

    if images.empty():
        return None

    for img in images.items():
        for channel_name in img.get_channels():
            channel, opts = img.get_channel(channel_name)
            if not opts["brightness"]:
                continue

            if channel_name in img.links["weight"]:
                weight_channel = img.links["weight"][channel_name]
                weight, _ = img.get_channel(weight_channel)
            else:
                weight = np.ones(channel.shape, dtype=np.float64)

            if channel_name not in summary:
                summary[channel_name] = deepcopy(channel.astype(np.float64))
                summary_weight[channel_name] = deepcopy(weight)
            else:

                try:
                    summary[channel_name] += channel
                    summary_weight[channel_name] += weight
                except Exception:
                    print("Can not add image. Skipping")

    result = vstarstack.library.data.DataFrame()
    for channel_name, channel in summary.items():
        print(channel_name)
        result.add_channel(channel, channel_name, brightness=True)
        result.add_channel(summary_weight[channel_name],
                           "weight-"+channel_name, weight=True)
        result.add_channel_link(channel_name, "weight-"+channel_name, "weight")

    return result
