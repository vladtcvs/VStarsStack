"""Remove stars"""
#
# Copyright (c) 2022 Vladislav Tsendrovskii
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

import scipy.signal
import numpy as np
import cv2

import vstarstack.library.stars.detect

def remove_stars(image : np.ndarray):
    """Remove stars from image"""
    size = 31
    stars = vstarstack.library.stars.detect.detect_stars(image)
    mask = np.zeros(image.shape)
    for star in stars:
        x = int(star["x"]+0.5)
        y = int(star["y"]+0.5)
        r = int(star["radius"]+0.5)
        cv2.circle(mask, (x,y), r, 1, -1)

    idx = (mask == 0)
    sidx = (mask != 0)
    nimg = np.zeros(image.shape)
    nimg[idx] = image[idx]
    nimg[sidx] = np.average(image[idx])
    filtered = scipy.signal.medfilt2d(nimg, size)
    nimg[sidx] = filtered[sidx]
    return nimg
