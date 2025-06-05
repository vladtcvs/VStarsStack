"""Mean filter with allowing to have NaN values"""
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
import scipy
import scipy.ndimage

import vstarstack.library.data

def nanmean_filter(image : np.ndarray,
                   radius : int):
    """nanmean filter"""
    fpsize=2*int(radius)+1
    filtered = scipy.ndimage.generic_filter(image, np.nanmean, [fpsize, fpsize],
                                            mode='constant', cval=np.nan)
    idx = np.where(np.isfinite(image))
    filtered[idx] = image[idx]
    return filtered

def filter_df(df : vstarstack.library.data.DataFrame, radius : int, in_place : bool = False):
    """nanmean filter to DF"""
    if in_place:
        fixed = df
    else:
        fixed = vstarstack.library.data.DataFrame(df.params)
    for channel in df.get_channels():
        layer, opts = df.get_channel(channel)
        if df.get_channel_option(channel, "encoded"):
            continue
        if df.get_channel_option(channel, "weight"):
            continue
        layer = nanmean_filter(layer, radius)
        fixed.add_channel(layer, channel, **opts)
    return fixed
