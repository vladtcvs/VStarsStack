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
from vstarstack.library.data import DataFrame
import numpy as np

def get_profile(image : np.ndarray):
    y = int(image.shape[0]/2)
    x = int(image.shape[1]/2)
    profile1 = image[y,:]
    profile2 = image[:,x]
    return (profile1 + profile2)/2

def get_star(image : np.ndarray, x : int, y : int, r : int):
    w = image.shape[1]
    h = image.shape[0]
    x1 = x - r
    x2 = x + r + 1
    y1 = y - r
    y2 = y + r + 1
    if x < r or y < r:
        return None
    if x > w-r-1 or y > h-r-1:
        return None
    return image[y1:y2, x1:x2]

def _interpolate(v1 : float, v2 : float, v : float):
    if v1 > v:
        return 0
    if v2 < v:
        return 1
    return (v - v1)/(v2-v1)

def get_width2(profile : np.ndarray):
    size = profile.shape[0]
    part = int(size/8)
    left = profile[0:part]
    right = profile[size-part-1:size-1]
    background = np.median(list(left) + list(right))
    noback = profile - background

    maxv = np.amax(noback)
    center = int(size/2)
    for i in range(size):
        if noback[i] > noback[center]:
            center = i
    x1 = center
    while True:
        val = noback[x1]
        if val <= maxv/2:
            break
        x1 -= 1
        if x1 == 0:
            break

    x2 = center
    while True:
        val = noback[x2]
        if val <= maxv/2:
            break
        x2 += 1
        if x2 == size-1:
            break

    vl1 = noback[x1]
    vl2 = noback[x1+1]

    vr1 = noback[x2]
    vr2 = noback[x2-1]

    d = _interpolate(vl1, vl2, maxv/2)
    x1 += d
    
    d = _interpolate(vr1, vr2, maxv/2)
    x2 -= d

    return x2 - x1

def find_fwhm(image : np.ndarray, x : int, y : int, radius : int) -> float:
    maxv = np.amax(image)
    si = get_star(image, x, y, radius*8+1)
    if si is None:
        return None
    if np.amax(si) == maxv:
        return None
    profile = get_profile(si)
    width = get_width2(profile)
    return width

def find_fwhm_df(image : DataFrame, x : int, y : int, radius : int) -> Tuple[dict,float]:
    channels = image.get_channels()
    widths = {}
    for channel in channels:
        layer, opts = image.get_channel(channel)
        if not opts["brightness"]:
            continue
        widths[channel] = find_fwhm(layer, x, y, radius)
    mean = np.median(list(widths.values()))
    return widths, mean
