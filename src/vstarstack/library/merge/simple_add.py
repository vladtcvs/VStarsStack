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

import logging
import numpy as np
import vstarstack.library.data
import vstarstack.library.common
import vstarstack.library.image_process.normalize
from vstarstack.library.data import DataFrame
from copy import deepcopy

logger = logging.getLogger(__name__)

def simple_add(images : vstarstack.library.common.IImageSource, ignore_saturated : bool = False) -> DataFrame:
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

            if ignore_saturated:
                saturated, _, _ = img.get_linked_channel(channel_name, "saturation")
                if saturated is not None:
                    mask = (saturated == 0).astype(np.uint)
                    weight = weight * mask
                    channel = channel * mask

            if channel_name not in summary:
                summary[channel_name] = deepcopy(channel.astype(np.float64))
                summary_weight[channel_name] = deepcopy(weight)
            else:
                try:
                    summary[channel_name] += channel
                    summary_weight[channel_name] += weight
                except Exception as excp:
                    logger.error(f"Can not add image. Skipping. Error = {excp}")

    result = vstarstack.library.data.DataFrame(params=params)
    for channel_name, channel in summary.items():
        logger.info(channel_name)
        weight_channel_name = "weight-"+channel_name
        result.add_channel(channel, channel_name, **channel_opts[channel_name])
        result.add_channel(summary_weight[channel_name], weight_channel_name, weight=True)
        result.add_channel_link(channel_name, weight_channel_name, "weight")

    return result
