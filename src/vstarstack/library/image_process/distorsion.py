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

import math
import numpy as np

import vstarstack.library.data
import vstarstack.library.common
import vstarstack.library.projection.perspective
import vstarstack.library.projection.tools

class Distorsion:
    """Distorsion description"""
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

def find_source_angle(target_angle : float,
                      distorsion : Distorsion):
    """Find source angle before distorsion"""
    k = distorsion.a * target_angle**2 + distorsion.b * target_angle + distorsion.c
    source_angle = target_angle * k
    return source_angle

def find_source_lonlat(target_lon : float,
                       target_lat : float,
                       distorsion : Distorsion):
    """Find source lon, lat befor distorsion"""
    target_angle = math.acos(math.cos(target_lat)*math.cos(target_lon))
    source_angle = find_source_angle(target_angle, distorsion)
    if source_angle < 1e-12:
        source_lat = target_lat
        source_lon = target_lon
    else:
        svec = np.array([math.cos(target_lat) * math.sin(target_lon), math.sin(target_lat)])
        svec /= (svec[0]**2+svec[1]**2)**0.5
        source_lat = math.asin(svec[1] * math.sin(source_angle))
        source_lon = math.atan2(svec[0] * math.sin(source_angle), math.cos(source_angle))
    return source_lon, source_lat

def _fix_distorsion(image : np.ndarray,
                    image_weight_layer : np.ndarray,
                    proj,
                    distorsion : Distorsion):

    h = image.shape[0]
    w = image.shape[1]

    fixed = np.zeros((h, w))
    fixed_weight = np.zeros((h, w))

    for y in range(h):
        for x in range(w):
            target_lat, target_lon = proj.project(y, x)
            lon, lat = find_source_lonlat(target_lon, target_lat, distorsion)
            source_y, source_x = proj.reverse(lat, lon)

            _, pixel = vstarstack.library.common.getpixel(image, source_y, source_x, False)
            _, pixel_weight = vstarstack.library.common.getpixel(
                image_weight_layer, source_y, source_x, False)

            fixed[y][x] = pixel
            fixed_weight[y][x] = pixel_weight

    return fixed, fixed_weight

def fix_distorsion(dataframe : vstarstack.library.data.DataFrame,
                   distorsion : Distorsion):
    """Fix image distorsion"""

    proj = vstarstack.library.projection.tools.get_projection(dataframe)
    if proj is None:
        return dataframe

    for channel in dataframe.get_channels():
        image, opts = dataframe.get_channel(channel)
        if opts["encoded"]:
            continue
        if opts["weight"]:
            continue
        weight_channel = dataframe.links["weight"][channel]
        image_weight, _ = dataframe.get_channel(weight_channel)

        fixed, fixed_weight = _fix_distorsion(image, image_weight, proj, distorsion)
        dataframe.add_channel(fixed, channel, **opts)
        dataframe.add_channel(fixed_weight, weight_channel, weight=True)
        dataframe.add_channel_link(channel, weight_channel, "weight")

    return dataframe
