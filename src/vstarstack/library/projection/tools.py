"""Add projection description to dataframe"""
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

import vstarstack.library.data
import vstarstack.library.projection.projections

def add_description(dataframe : vstarstack.library.data.DataFrame, projection : str, **argv):
    """Add projection description to dataframe"""
    if projection == "perspective":
        F = argv["F"]
        kw = argv["kw"]
        kh = argv["kh"]
        dataframe.add_parameter(F, "perspective_F")
        dataframe.add_parameter(kw, "perspective_kw")
        dataframe.add_parameter(kh, "perspective_kh")
    elif projection == "equirectangular":
        pass
    elif projection == "orthographic":
        a = argv["a"]
        b = argv["b"]
        angle = argv["angle"]
        rot = argv["rot"]
        dataframe.add_parameter(a, "orthographic_a")
        dataframe.add_parameter(b, "orthographic_b")
        dataframe.add_parameter(angle, "orthographic_angle")
        dataframe.add_parameter(rot, "orthographic_rot")

    dataframe.add_parameter(projection, "projection")

def get_projection(dataframe : vstarstack.library.data.DataFrame):
    """Get projection from dataframe"""
    if "projection" not in dataframe.params:
        return None
    if dataframe.params["projection"] == "perspective":
        F = dataframe.params["perspective_F"]
        w = dataframe.params["w"]
        h = dataframe.params["h"]
        W = w * dataframe.params["perspective_kw"]
        H = h * dataframe.params["perspective_kh"]
        return vstarstack.library.projection.projections.PerspectiveProjection(w, h, W, H, F)

    if dataframe.params["projection"] == "equirectangular":
        w = dataframe.params["w"]
        h = dataframe.params["h"]
        return vstarstack.library.projection.projections.EquirectangularProjection(w, h)
    
    if dataframe.params["projection"] == "orthographic":
        w = dataframe.params["w"]
        h = dataframe.params["h"]
        a = dataframe.params["a"]
        b = dataframe.params["b"]
        angle = dataframe.params["angle"]
        rot = dataframe.params["rot"]
        return vstarstack.library.projection.projections.OrthographicProjection(w, h, a, b, angle, rot)

    raise Exception("Unknown projection")
