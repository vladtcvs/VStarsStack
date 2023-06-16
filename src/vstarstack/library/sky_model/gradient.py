"""Gradient sky model"""
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

import numpy as np

import vstarstack.library.sky_model.remove_stars


def model(image):
    """Build Gradient sky model"""
    shape = image.shape
    w = shape[1]
    h = shape[0]

    corner_w = int(w/10)
    corner_h = int(h/10)

    image_nostars = vstarstack.library.sky_model.remove_stars.remove_stars(image)

    left_top = np.average(image_nostars[0:corner_h, 0:corner_w])
    left_bottom = np.average(image_nostars[h-1-corner_h:h-1, 0:corner_w])

    right_top = np.average(image_nostars[0:corner_h, w-1-corner_w:w-1])
    right_bottom = np.average(
        image_nostars[h-1-corner_h:h-1, w-1-corner_w:w-1])

    bottom_k = np.array(range(h))/(h-1)
    top_k = 1-bottom_k

    right_k = np.array(range(w))/(w-1)
    left_k = 1-right_k

    left_top_k = top_k[:, np.newaxis] * left_k
    right_top_k = top_k[:, np.newaxis] * right_k
    left_bottom_k = bottom_k[:, np.newaxis] * left_k
    right_bottom_k = bottom_k[:, np.newaxis] * right_k

    sky = left_top * left_top_k + left_bottom * left_bottom_k + \
        right_top * right_top_k + right_bottom * right_bottom_k
    return sky
