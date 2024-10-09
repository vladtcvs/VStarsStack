"""Convert dataframe to brightness"""
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

from typing import Tuple
import numpy as np
import scipy.ndimage

import vstarstack.library.data
import vstarstack.library.common
import vstarstack.tool
import vstarstack.tool.cfg

def df_to_gray(df : vstarstack.library.data.DataFrame, interpolate_cfa : bool | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """convert dataframe to brightness array"""
    df = vstarstack.library.common.df_weight_0_to_0(df, False)

    gray = None
    total_weight = None
    for channel in df.get_channels():
        layer, opts = df.get_channel(channel)
        if not opts["brightness"]:
            continue

        weight,_,_ = df.get_linked_channel(channel, "weight")
        if weight is None:
            weight = np.ones(layer.shape)

        idx = np.where(weight == 0)
        layer = layer / weight
        weight[np.where(weight != 0)] = 1
        layer[idx] = 0

        if interpolate_cfa is None:
            interpolate_cfa = df.get_channel_option(channel, "cfa")

        if interpolate_cfa:
            h = layer.shape[0]
            w = layer.shape[1]
            shifted = np.zeros(layer.shape)
            weights = np.zeros(layer.shape)

            shifted[0:h-1,0:w] += layer[1:h,0:w]
            weights[0:h-1,0:w] += weight[1:h,0:w]
            
            shifted[0:h-1,0:w-1] += layer[1:h,1:w]
            weights[0:h-1,0:w-1] += weight[1:h,1:w]
            
            shifted[0:h,0:w-1] += layer[0:h,1:w]
            weights[0:h,0:w-1] += weight[0:h,1:w]

            shifted[1:h,0:w-1] += layer[0:h-1,1:w]
            weights[1:h,0:w-1] += weight[0:h-1,1:w]

            shifted[1:h,0:w] += layer[0:h-1,0:w]
            weights[1:h,0:w] += weight[0:h-1,0:w]

            shifted[0:h,1:w] += layer[0:h,0:w-1]
            weights[0:h,1:w] += weight[0:h,0:w-1]

            shifted[1:h,1:w] += layer[0:h-1,0:w-1]
            weights[1:h,1:w] += weight[0:h-1,0:w-1]

            shifted[0:h-1,1:w] += layer[1:h,0:w-1]
            weights[0:h-1,1:w] += weight[1:h,0:w-1]

            shifted /= weights
            shifted[np.where(weights == 0)] = 0
            weights[np.where(weights != 0)] = 1
            layer[idx] = shifted[idx]
            weight[idx] = weights[idx]

        if gray is None:
            gray = layer
            if weight is not None:
                total_weight = weight
        else:
            gray = gray + layer
            if weight is not None:
                total_weight = total_weight + weight

    if total_weight is not None:
        gray = gray / total_weight
        gray[np.where(total_weight == 0)] = 0
        total_weight[np.where(total_weight != 0)] = 1

    return gray, total_weight
