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

import imageio
import numpy as np

import vstarstack.library.stars.detect as detect

def test_1():
    """Try detector on stars1.png"""
    image = imageio.imread('stars2.png')[:,:,0:3]
    image = image / np.amax(image)
    gray = image[:,:,0]
    stars = detect.detect_stars(gray)
