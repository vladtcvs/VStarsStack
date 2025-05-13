#
# Copyright (c) 2025 Vladislav Tsendrovskii
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

import cv2
import numpy as np
from vstarstack.library.data import DataFrame

def star_magnitude_summ(image : np.ndarray, x : int, y : int, radius : int):
    """Find sum of pixels in circle at x,y"""
    area = image[y-radius:y+radius+1, x-radius:x+radius+1]
    if area.shape[0] == 0 or area.shape[1] == 0:
        return None, None, None
    mask = np.zeros((2*radius+1, 2*radius+1))
    cv2.circle(mask, (radius, radius), radius, 1, -1)
    area = area * mask
    maxv = np.amax(area)
    return np.sum(area), int(np.sum(mask)), maxv

def star_magnitude_summ_df(image : DataFrame, x : int, y : int, radius : int):
    """Find sum of pixels in circle at x,y"""
    vals = {}
    for cn in image.get_channels():
        channel, opts = image.get_channel(cn)
        if opts["brightness"]:
            vals[cn] = star_magnitude_summ(channel, x, y, radius)
    return vals

def star_magnitude_summ_nobg(image : np.ndarray, x : int, y : int, radius : int):
    """Find sum of pixels in circle at x,y"""
    mr = int(radius * 3)
    area = image[y-mr:y+mr+1, x-mr:x+mr+1]
    if area.shape[0] == 0 or area.shape[1] == 0:
        return None, None, None, None
    mask = np.zeros(area.shape)
    cv2.circle(mask, (mr, mr), mr, 1, -1)
    cv2.circle(mask, (mr, mr), radius*2, 0, -1)
    area = area * mask
    bg = np.sum(area) / np.sum(mask)

    area = image[y-radius:y+radius+1, x-radius:x+radius+1]
    
    mask = np.zeros(area.shape)
    cv2.circle(mask, (radius, radius), radius, 1, -1)
    area = area * mask
    maxv = np.amax(area)
    area = area - bg * mask

    return np.sum(area), int(np.sum(mask)), bg, maxv

def star_magnitude_summ_nobg_df(image : DataFrame, x : int, y : int, radius : int):
    """Find sum of pixels in circle at x,y"""
    vals = {}
    for cn in image.get_channels():
        channel, opts = image.get_channel(cn)
        if opts["brightness"]:
            vals[cn] = star_magnitude_summ_nobg(channel, x, y, radius)
    return vals
