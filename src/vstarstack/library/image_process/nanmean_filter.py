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

def nanmean_filter(image : np.ndarray,
                   radius : int):
    """nanmean filter"""
    result = np.zeros(image.shape)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            left = max(0, x-radius)
            top  = max(0, y-radius)
            right = min(image.shape[1], x+radius)
            bottom = min(image.shape[0], y+radius)
            subimage = image[top:bottom,left:right]
            value = np.nanmean(subimage)
            result[y,x] = value

    return result
