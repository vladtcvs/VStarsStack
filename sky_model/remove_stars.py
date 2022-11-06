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

import targets.stars.detect
import scipy.signal
import numpy as np

def remove_stars(image):
    size = 31
    _,mask = targets.stars.detect.detect(image)
    idx = (mask == 0)
    sidx = (mask != 0)
    nimg = np.zeros(image.shape)
    nimg[idx] = image[idx]
    nimg[sidx] = np.average(image[idx])
    filtered = scipy.signal.medfilt2d(nimg, size)
    nimg[sidx] = filtered[sidx]
    return nimg
