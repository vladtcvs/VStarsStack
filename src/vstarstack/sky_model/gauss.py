"""Gauss sky model"""
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
import cv2

import vstarstack.usage
import vstarstack.common
import vstarstack.targets.stars.detect
import vstarstack.cfg


def model(project, image):
    """Build Gaussian sky model"""
    _, mask = vstarstack.targets.stars.detect.detect(project, image)
    shape = image.shape
    sky_blur = int(shape[1]/4)*2+1

    sky = np.zeros(shape)
    idx = (mask == 0)
    nidx = (mask != 0)
    sky[idx] = image[idx]
    average = np.mean(sky, axis=(0, 1))
    sky[nidx] = average
    sky = cv2.GaussianBlur(sky, (sky_blur, sky_blur), 0)
    return sky
