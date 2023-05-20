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

import math
import numpy as np

import vstarstack.library.common
import vstarstack.library.data
from vstarstack.library.projection.orthographic import Projection as PlanetProjection
from vstarstack.library.projection.equirectangular import Projection as MapProjection

def build_surface_map(image : vstarstack.library.data.DataFrame,
                      horizontal_size : float,
                      vertical_size : float,
                      angle : float,
                      rot : float,
                      maph : int):
    """Build surface map of the planet"""
    w = image.shape[1]
    h = image.shape[0]
    surface = np.zeros((maph, 2*maph))
    surface_proj = MapProjection(2*maph, maph)
    planet_proj = PlanetProjection(w, h, horizontal_size, vertical_size, angle, rot)
    mask = np.zeros((maph, 2*maph))
    for y in range(maph):
        for x in range(2*maph):
            lat, lon = surface_proj.reverse(y, x)
            if lon > math.pi/2 and lon < 3*math.pi/2:
                continue
            X, Y = planet_proj.reverse(lon, lat)
            res, pix = vstarstack.library.common.getpixel(image, Y, X)
            if res:
                surface[y, x] = pix
                mask[y, x] = 1
    return surface, mask
