"""Add projection description to dataframe"""
#
# Copyright (c) 2023-2024 Vladislav Tsendrovskii
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

import typing

import vstarstack.library.data
from vstarstack.library.projection import ProjectionType
from vstarstack.library.projection.projections import PerspectiveProjection
from vstarstack.library.projection.projections import EquirectangularProjection
from vstarstack.library.projection.projections import OrthographicProjection

def add_description(dataframe : vstarstack.library.data.DataFrame, projection : ProjectionType, **argv):
    """Add projection description to dataframe"""
    if projection == ProjectionType.NoneProjection:
        name = "none"
    elif projection == ProjectionType.Perspective:
        F = argv["F"]
        kw = argv["kw"]
        kh = argv["kh"]
        dataframe.add_parameter(F, "projection_perspective_F")
        dataframe.add_parameter(kw, "projection_perspective_kw")
        dataframe.add_parameter(kh, "projection_perspective_kh")
        name = "perspective"
    elif projection == ProjectionType.Equirectangular:
        name = "equirectangular"
    elif projection == ProjectionType.Orthographic:
        a = argv["a"]
        b = argv["b"]
        angle = argv["angle"]
        rot = argv["rot"]
        dataframe.add_parameter(a, "projection_orthographic_a")
        dataframe.add_parameter(b, "projection_orthographic_b")
        dataframe.add_parameter(angle, "projection_orthographic_angle")
        dataframe.add_parameter(rot, "projection_orthographic_rot")
        name = "orthographic"
    else:
        return
    dataframe.add_parameter(name, "projection")

def extract_description(dataframe : vstarstack.library.data.DataFrame) -> typing.Tuple[ProjectionType, dict]:
    """Extract projection description from dataframe"""
    projection = dataframe.get_parameter("projection")
    if projection is None:
        return None, {}
    elif projection == "none":
        return ProjectionType.NoneProjection, {}
    elif projection == "perspective":
        return ProjectionType.Perspective, {
            "F" : dataframe.get_parameter("projection_perspective_F"),
            "kw" : dataframe.get_parameter("projection_perspective_kw"),
            "kh" : dataframe.get_parameter("projection_perspective_kh"),
        }
    elif projection == "equirectangular":
        return ProjectionType.Equirectangular, {}
    elif projection == "orthographic":
        return ProjectionType.Orthographic, {
            "a" : dataframe.get_parameter("projection_orthographic_a"),
            "b" : dataframe.get_parameter("projection_orthographic_b"),
            "angle" : dataframe.get_parameter("projection_orthographic_angle"),
            "rot" : dataframe.get_parameter("projection_orthographic_rot"),
        }
    else:
        raise Exception("Unknown projection")

def build_projection(projection: ProjectionType, desc : dict, shape : tuple):
    """Build projection by description"""
    w = shape[1]
    h = shape[0]

    if projection == ProjectionType.NoneProjection:
        raise Exception("Trying to build \"NoneProjection\"")

    elif projection == ProjectionType.Perspective:
        F = desc["F"]
        kw = desc["kw"]
        kh = desc["kh"]
        return PerspectiveProjection(w, h, kw, kh, F)

    elif projection == ProjectionType.Equirectangular:
        return EquirectangularProjection(w, h)

    elif projection == ProjectionType.Orthographic:
        a = desc["a"]
        b = desc["b"]
        angle = desc["angle"]
        rot = desc["rot"]
        return OrthographicProjection(w, h, a, b, angle, rot)

    else:
        raise Exception("Trying to build some unknown projection")

def get_projection(dataframe : vstarstack.library.data.DataFrame):
    """Get projection from dataframe"""
    projection, desc = extract_description(dataframe)
    shape = (dataframe.get_parameter("h"), dataframe.get_parameter("w"))
    return build_projection(projection, desc, shape)
