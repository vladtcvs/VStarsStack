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

from typing import Tuple
import numpy as np
import math

def find_star_profile(image : np.ndarray, npoints : int) -> Tuple[float, np.ndarray]:
    """Find star profile"""
    w = image.shape[1]
    h = image.shape[0]
    cx = int(w/2)
    cy = int(h/2)
    profile = np.zeros((w*h, 2))
    for y in range(h):
        dy = y - cy
        for x in range(w):
            dx = x - cx
            r = math.sqrt(dx**2+dy**2)
            profile[y*w+x, 0] = r
            profile[y*w+x, 1] = image[y,x]
    rmax = np.max(profile[:,0])
    smooth_profile = np.zeros((npoints,))
    for i in range(npoints):
        r_1 = rmax * i / npoints
        r_2 = rmax * (i+1) / npoints

        idxs = np.where(np.logical_and(profile[:,0] >= r_1, profile[:,0] <= r_2))
        smooth_profile[i] = np.mean(profile[idxs,1])

    return rmax, smooth_profile

def restore_star_profile(profile : np.ndarray) -> np.ndarray:
    """If star profile has saturation, this function restores center"""
    